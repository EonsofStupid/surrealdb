use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;

use crate::catalog::{DatabaseId, IndexId, NamespaceId, QortexParams, TableId};
use crate::ctx::FrozenContext;
use crate::idx::IndexKeyBase;
use crate::idx::trees::qortex::index::QortexIndex;

/// A thread-safe, shared reference to a [`QortexIndex`].
///
/// The `QortexIndex` manages its own internal concurrency via `RwLock` fields,
/// so the outer `Arc` provides shared ownership without additional locking.
pub(crate) type SharedQortexIndex = Arc<QortexIndex>;

pub(crate) type SharedQortexKey = (NamespaceId, DatabaseId, TableId, IndexId);

/// Registry of all active QORTEX indexes, keyed by `(NamespaceId, DatabaseId, TableId, IndexId)`.
///
/// Indexes are lazily initialized on first access and cached for subsequent use.
pub(crate) struct QortexIndexes(Arc<RwLock<HashMap<SharedQortexKey, SharedQortexIndex>>>);

impl Default for QortexIndexes {
	fn default() -> Self {
		Self(Arc::new(RwLock::new(HashMap::new())))
	}
}

impl QortexIndexes {
	/// Retrieves or lazily creates a QORTEX index for the given table and index key.
	///
	/// Uses a double-checked locking pattern: first attempts a read lock lookup,
	/// then falls back to a write lock for creation if the index is not yet cached.
	pub(super) async fn get(
		&self,
		ctx: &FrozenContext,
		tb: TableId,
		ikb: &IndexKeyBase,
		p: &QortexParams,
	) -> Result<SharedQortexIndex> {
		let key = (ikb.ns(), ikb.db(), tb, ikb.index());
		let h = self.0.read().await.get(&key).cloned();
		if let Some(h) = h {
			return Ok(h);
		}
		let mut w = self.0.write().await;
		let ix = match w.entry(key) {
			Entry::Occupied(e) => Arc::clone(e.get()),
			Entry::Vacant(e) => {
				let h = Arc::new(
					QortexIndex::new(
						ctx.get_index_stores().vector_cache().clone(),
						&ctx.tx(),
						ikb.clone(),
						tb,
						p,
					)
					.await?,
				);
				e.insert(Arc::clone(&h));
				h
			}
		};
		Ok(ix)
	}

	/// Removes a QORTEX index from the registry.
	pub(super) async fn remove(&self, tb: TableId, ikb: &IndexKeyBase) -> Result<()> {
		let key = (ikb.ns(), ikb.db(), tb, ikb.index());
		self.0.write().await.remove(&key);
		Ok(())
	}
}
