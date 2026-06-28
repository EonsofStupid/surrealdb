//! Qortex index — embeds a Qdrant `segment` (the qortex engine) as a vector-index backing.
//!
//! Fusion path B, Increment 1: a clean wrapper proven by a REAL knn test, BEFORE wiring it into
//! SurrealDB's index layer (catalog `Index::Qortex`, planner/executor dispatch, `DEFINE INDEX … QORTEX`).
//! Qdrant's HNSW owns its storage (id_tracker/vector_storage/quantized/payload), so we embed the whole
//! Segment rather than the bare graph — see `idx/trees/diskann` for the BYO-provider contrast.
//!
//! Uses the production constructor `build_segment` (not the `testing`-gated `build_simple_segment`).
//! Inc 1 uses a `Plain` (exact) index to prove the embed + knn; Inc 2+ swaps in `Indexes::Hnsw`.

use std::collections::HashMap;
use std::path::Path;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::counter::hardware_counter::HardwareCounterCell;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry as _, SegmentEntry as _};
use segment::segment::Segment;
use segment::segment_constructor::build_segment;
use segment::types::{
	Distance, Indexes, PayloadStorageType, PointIdType, SegmentConfig, VectorDataConfig,
	VectorStorageType, WithPayload,
};

/// A vector index backed by an embedded Qdrant segment (the qortex engine).
pub(crate) struct QortexIndex {
	segment: Segment,
	/// Monotonic op number Qdrant requires per write (its WAL sequence).
	op: u64,
}

impl QortexIndex {
	/// Open (or create) a qortex-backed vector index at `path` for `dim`-d vectors.
	pub(crate) fn open(path: &Path, dim: usize, distance: Distance) -> anyhow::Result<Self> {
		let config = SegmentConfig {
			vector_data: HashMap::from([(
				DEFAULT_VECTOR_NAME.to_owned(),
				VectorDataConfig {
					size: dim,
					distance,
					storage_type: VectorStorageType::InRamChunkedMmap,
					index: Indexes::Plain {},
					quantization_config: None,
					multivector_config: None,
					datatype: None,
				},
			)]),
			sparse_vector_data: Default::default(),
			payload_storage_type: PayloadStorageType::InRamMmap,
		};
		Ok(Self {
			segment: build_segment(path, &config, None, true)?,
			op: 0,
		})
	}

	/// Insert/replace a vector by id.
	pub(crate) fn insert(&mut self, id: u64, vector: &[f32]) -> anyhow::Result<()> {
		self.op += 1;
		let hw = HardwareCounterCell::disposable();
		self.segment.upsert_point(self.op, id.into(), only_default_vector(vector), &hw)?;
		Ok(())
	}

	/// k-nearest-neighbours for `vector` → (point id, score), best first.
	pub(crate) fn search(&self, vector: &[f32], k: usize) -> anyhow::Result<Vec<(PointIdType, f32)>> {
		let q: QueryVector = vector.to_vec().into();
		let query_context = QueryContext::new(20_000, HwMeasurementAcc::disposable());
		let sqc = query_context.get_segment_query_context();
		let batches = self.segment.search_batch(
			DEFAULT_VECTOR_NAME,
			&[&q],
			&WithPayload::default(),
			&false.into(),
			None,
			k,
			None,
			&sqc,
		)?;
		let res = batches.into_iter().next().unwrap_or_default();
		Ok(res.into_iter().map(|p| (p.id, p.score)).collect())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn qortex_segment_knn_smoke() {
		let dir = std::env::temp_dir().join(format!("qortex_inc1_{}", std::process::id()));
		let _ = std::fs::remove_dir_all(&dir);
		std::fs::create_dir_all(&dir).unwrap();

		let mut idx = QortexIndex::open(&dir, 4, Distance::Cosine).unwrap();
		idx.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
		idx.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
		idx.insert(3, &[0.95, 0.05, 0.0, 0.0]).unwrap();

		let res = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
		eprintln!("qortex knn → {res:?}");
		assert!(!res.is_empty(), "knn returned no results");
		// nearest to [1,0,0,0] must be point 1 (the exact match), not point 2
		assert_eq!(res[0].0, 1u64.into(), "wrong nearest neighbour");

		let _ = std::fs::remove_dir_all(&dir);
	}
}
