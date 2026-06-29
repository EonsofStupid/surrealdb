//! Store-level QORTEX index: a KV-backed wrapper around an embedded qortex segment.
//!
//! Storage contract (Fusion path B, Increment 3a):
//! - The KV store is the source of truth. Three key families back the index:
//!   `!qs` (next point-id counter), `!qi` (record id → point-id), and `!qv`
//!   (point-id → record id + vector payload).
//! - The embedded qortex [`QortexSegment`] is an ephemeral in-RAM mirror that is
//!   rebuilt from the `!qv` range whenever it is stale (see [`QortexIndex::check_state`]).
//! - Writes go straight to KV ([`QortexIndex::index`]) and mark the segment dirty,
//!   so the next read rebuilds from committed KV. This keeps the segment correct
//!   across transaction rollbacks without an async pending/compaction pipeline.
//!
//! Deferred to Inc 3b (explicitly): async pending/compaction, and filtered knn.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};
use half::f16;
use reblessive::tree::Stk;
use revision::revisioned;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::catalog::{Distance, QortexParams, TableId, VectorType};
use crate::ctx::{Context, FrozenContext};
use crate::dbs::Options;
use crate::expr::Cond;
use crate::fnc::util::math::ToFloat;
use crate::idx::IndexKeyBase;
use crate::idx::planner::ScanDirection;
use crate::idx::planner::iterators::KnnIteratorResult;
use crate::idx::trees::hnsw::cache::VectorCache;
use crate::idx::trees::qortex::QortexSegment;
use crate::idx::trees::vector::{SerializedVector, Vector};
use crate::key::index::qv::Qv;
use crate::kvs::{KVValue, NORMAL_BATCH_SIZE, ScanLimit, Transaction, impl_kv_value_revisioned};
use crate::val::{Number, RecordId, RecordIdKey, Value};
use std::sync::Arc;

/// Persisted QORTEX point payload: the owning record id plus its `f32` vector.
///
/// Stored under the record's `!qv{point_id}` key. The segment is rebuilt from
/// these payloads, so this is the durable representation of every indexed point.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct QortexVecValue {
	/// The record id this point belongs to.
	pub(crate) rid: RecordIdKey,
	/// The indexed vector, stored as `f32` (the qortex segment representation).
	pub(crate) vector: Vec<f32>,
}

impl_kv_value_revisioned!(QortexVecValue);

/// A vector index backed by an embedded qortex segment, with KV as source of truth.
pub(crate) struct QortexIndex {
	/// Expected vector dimensionality.
	dim: usize,
	/// Catalog distance metric (validated/mapped to the segment metric on open).
	distance: Distance,
	/// The declared vector type used to coerce/validate document values.
	vector_type: VectorType,
	/// Stable table id (reserved for Inc 3b filtered knn / cache scoping).
	#[allow(dead_code)]
	table_id: TableId,
	/// Key base for generating index-related storage keys.
	ikb: IndexKeyBase,
	/// Shared vector cache (reserved for Inc 3b filtered knn).
	#[allow(dead_code)]
	vector_cache: VectorCache,
	/// The in-RAM qortex segment, rebuilt from KV on demand.
	segment: RwLock<QortexSegment>,
	/// Filesystem directory currently backing the segment (cleaned on rebuild).
	seg_dir: std::sync::Mutex<PathBuf>,
	/// Whether the segment reflects the latest committed KV state.
	loaded: AtomicBool,
}

impl QortexIndex {
	/// Creates a new QORTEX index handle with an empty, not-yet-loaded segment.
	///
	/// The segment is loaded lazily on the first [`check_state`](Self::check_state).
	pub(crate) async fn new(
		vector_cache: VectorCache,
		_tx: &Transaction,
		ikb: IndexKeyBase,
		tb: TableId,
		p: &QortexParams,
	) -> Result<Self> {
		let seg_distance = Self::map_distance(&p.distance)?;
		let dir = Self::unique_dir();
		let segment = QortexSegment::open(&dir, p.dimension as usize, seg_distance)?;
		Ok(Self {
			dim: p.dimension as usize,
			distance: p.distance.clone(),
			vector_type: p.vector_type,
			table_id: tb,
			ikb,
			vector_cache,
			segment: RwLock::new(segment),
			seg_dir: std::sync::Mutex::new(dir),
			loaded: AtomicBool::new(false),
		})
	}

	/// Maps a catalog distance metric to the qortex (Qdrant) segment metric.
	fn map_distance(d: &Distance) -> Result<segment::types::Distance> {
		use segment::types::Distance as SegDistance;
		Ok(match d {
			Distance::Cosine | Distance::CosineNormalized => SegDistance::Cosine,
			Distance::Euclidean => SegDistance::Euclid,
			Distance::InnerProduct => SegDistance::Dot,
			Distance::Manhattan => SegDistance::Manhattan,
			_ => bail!(
				"QORTEX supports COSINE, COSINE_NORMALIZED, EUCLIDEAN, INNER_PRODUCT, and MANHATTAN distances"
			),
		})
	}

	/// Returns a unique temp directory to back one segment generation.
	fn unique_dir() -> PathBuf {
		std::env::temp_dir().join(format!("surreal_qortex_{}", Uuid::now_v7()))
	}

	/// Coerces and validates a document value into the segment's `f32` vector form.
	fn value_to_f32(&self, value: Value) -> Result<Vec<f32>> {
		let sv = SerializedVector::try_from_value(self.vector_type, self.dim, value)?;
		Vector::check_expected_dimension(sv.dimension(), self.dim)?;
		Ok(serialized_to_f32(&sv))
	}

	/// Picks the single indexed vector from the (possibly multi-valued) content.
	///
	/// Inc 3a indexes a single vector per record (the first non-nullish value).
	fn first_vector(&self, values: Vec<Value>) -> Result<Option<Vec<f32>>> {
		for value in values.into_iter().filter(|v| !v.is_nullish()) {
			// TODO(Inc 3b): multi-valued vector columns (index every value).
			return Ok(Some(self.value_to_f32(value)?));
		}
		Ok(None)
	}

	/// Applies one record change directly to KV and marks the segment dirty.
	///
	/// KV is authoritative; the in-RAM segment is rebuilt from KV on the next
	/// read. This indexes synchronously (no async pending/compaction in 3a).
	pub(crate) async fn index(
		&self,
		ctx: &Context,
		id: &RecordIdKey,
		old_values: Option<Vec<Value>>,
		new_values: Option<Vec<Value>>,
	) -> Result<()> {
		if old_values.is_none() && new_values.is_none() {
			return Ok(());
		}
		let tx = ctx.tx();
		// Resolve the desired new vector (if any).
		let new_vector = match new_values {
			Some(values) => self.first_vector(values)?,
			None => None,
		};
		match new_vector {
			// Add or update: allocate (or reuse) a point-id and write the payload.
			Some(vector) => {
				let qi_key = self.ikb.new_qi_key(id);
				let pid = match tx.get(&qi_key, None).await? {
					Some(pid) => pid,
					None => {
						let qs_key = self.ikb.new_qs_key();
						let next = tx.get(&qs_key, None).await?.unwrap_or(0);
						tx.set(&qs_key, &next.saturating_add(1)).await?;
						next
					}
				};
				let payload = QortexVecValue {
					rid: id.clone(),
					vector,
				};
				tx.set(&self.ikb.new_qv_key(pid), &payload).await?;
				tx.set(&qi_key, &pid).await?;
			}
			// Removal: drop the point payload and the record→point-id mapping.
			None => {
				let qi_key = self.ikb.new_qi_key(id);
				if let Some(pid) = tx.get(&qi_key, None).await? {
					tx.del(&self.ikb.new_qv_key(pid)).await?;
					tx.del(&qi_key).await?;
				}
			}
		}
		// Mark the in-RAM segment stale so the next read rebuilds from KV.
		self.loaded.store(false, Ordering::Release);
		Ok(())
	}

	/// Ensures the in-RAM segment reflects the latest committed KV state.
	///
	/// Rebuilds the segment by scanning the `!qv` range when it is stale. Uses a
	/// double-check under the write lock so concurrent callers rebuild at most once.
	pub(crate) async fn check_state(&self, ctx: &FrozenContext) -> Result<()> {
		if self.loaded.load(Ordering::Acquire) {
			return Ok(());
		}
		// Build a fresh segment from KV outside the write lock (the scan is the
		// expensive part); swap it in under the lock with a double-check.
		let seg_distance = Self::map_distance(&self.distance)?;
		let new_dir = Self::unique_dir();
		let mut new_seg = QortexSegment::open(&new_dir, self.dim, seg_distance)?;
		let tx = ctx.tx();
		let rng = self.ikb.new_qv_range()?;
		let mut cursor = tx.open_vals_cursor(rng, ScanDirection::Forward, 0, None).await?;
		loop {
			let batch = cursor.next_batch(ScanLimit::Count(NORMAL_BATCH_SIZE)).await?;
			if batch.is_empty() {
				break;
			}
			for (key, value) in &batch {
				let qv = Qv::decode_key(key)?;
				let payload = QortexVecValue::kv_decode_value(value, ())?;
				new_seg.insert(qv.point_id, &payload.vector)?;
			}
		}
		drop(cursor);
		// Swap the rebuilt segment in, double-checking another task did not win.
		let mut guard = self.segment.write().await;
		if self.loaded.load(Ordering::Acquire) {
			drop(guard);
			let _ = std::fs::remove_dir_all(&new_dir);
			return Ok(());
		}
		*guard = new_seg;
		let old_dir = {
			let mut d =
				self.seg_dir.lock().map_err(|_| anyhow::anyhow!("qortex segment dir lock poisoned"))?;
			std::mem::replace(&mut *d, new_dir)
		};
		self.loaded.store(true, Ordering::Release);
		drop(guard);
		// The previous segment is dropped above; its files can now be removed.
		let _ = std::fs::remove_dir_all(&old_dir);
		Ok(())
	}

	/// Performs a k-nearest-neighbour search against the qortex segment.
	///
	/// Returns iterator-ready results: `(record id, score, None)`, best first.
	pub(crate) async fn knn_search(
		&self,
		ctx: &FrozenContext,
		stk: &mut Stk,
		pt: &[Number],
		k: usize,
		ef: usize,
		cond_filter: Option<(&Options, Arc<Cond>)>,
	) -> Result<VecDeque<KnnIteratorResult>> {
		// `ef`/`stk` are unused by the exact-index 3a path.
		let _ = (ef, stk);
		// TODO(Inc 3b): filtered knn — honour `cond_filter` instead of ignoring it.
		let _ = cond_filter;
		// Make sure the segment reflects committed KV state.
		self.check_state(ctx).await?;
		// Convert the query vector to `f32`.
		let query: Vec<f32> = pt.iter().map(|n| n.to_float() as f32).collect();
		let hits = {
			let guard = self.segment.read().await;
			guard.search(&query, k)?
		};
		let tx = ctx.tx();
		let mut res = VecDeque::with_capacity(hits.len());
		for (point_id, score) in hits {
			let segment::types::PointIdType::NumId(pid) = point_id else {
				// Inc 3a only ever stores NumId points; skip anything else.
				continue;
			};
			let Some(payload) = tx.get(&self.ikb.new_qv_key(pid), None).await? else {
				// Payload removed since the segment was built; skip stale hit.
				continue;
			};
			let record_id = RecordId::new(self.ikb.table().clone(), payload.rid);
			res.push_back((Arc::new(record_id), score as f64, None));
		}
		Ok(res)
	}
}

/// Converts a persisted serialized vector into the segment's `f32` representation.
fn serialized_to_f32(sv: &SerializedVector) -> Vec<f32> {
	match sv {
		SerializedVector::F64(v) => v.iter().map(|x| *x as f32).collect(),
		SerializedVector::F32(v) => v.clone(),
		SerializedVector::I64(v) => v.iter().map(|x| *x as f32).collect(),
		SerializedVector::I32(v) => v.iter().map(|x| *x as f32).collect(),
		SerializedVector::I16(v) => v.iter().map(|x| *x as f32).collect(),
		SerializedVector::F16(v) => v.iter().map(|x| f16::from_bits(*x).to_f32()).collect(),
		SerializedVector::I8(v) => v.iter().map(|x| *x as f32).collect(),
		SerializedVector::U8(v) => v.iter().map(|x| *x as f32).collect(),
	}
}
