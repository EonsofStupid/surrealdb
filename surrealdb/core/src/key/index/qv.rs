//! Stores the record id + vector payload for one QORTEX point-id.
//!
//! KV is the source of truth for the QORTEX index: the embedded qortex segment
//! is rebuilt in-RAM by scanning every `!qv` entry. Each key is suffixed by the
//! `u64` point-id; the value carries the owning record id and the `f32` vector.

use std::borrow::Cow;
use std::ops::Range;

use anyhow::Result;
use storekey::{BorrowDecode, Encode};

use crate::catalog::{DatabaseId, IndexId, NamespaceId};
use crate::idx::trees::qortex::index::QortexVecValue;
use crate::kvs::{KVKey, Key, impl_kv_key_storekey};
use crate::val::TableName;

/// Stores one QORTEX point payload (record id + vector) keyed by point-id.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Encode, BorrowDecode)]
#[storekey(format = "()")]
pub(crate) struct Qv<'a> {
	__: u8,
	_a: u8,
	pub ns: NamespaceId,
	_b: u8,
	pub db: DatabaseId,
	_c: u8,
	pub tb: Cow<'a, TableName>,
	_d: u8,
	pub ix: IndexId,
	_e: u8,
	_f: u8,
	_g: u8,
	pub point_id: u64,
}

impl_kv_key_storekey!(Qv<'_> => QortexVecValue);

impl<'a> Qv<'a> {
	/// Creates the `!qv{point_id}` key for one point payload.
	pub fn new(ns: NamespaceId, db: DatabaseId, tb: &'a TableName, ix: IndexId, point_id: u64) -> Self {
		Self {
			__: b'/',
			_a: b'*',
			ns,
			_b: b'*',
			db,
			_c: b'*',
			tb: Cow::Borrowed(tb),
			_d: b'+',
			ix,
			_e: b'!',
			_f: b'q',
			_g: b'v',
			point_id,
		}
	}

	/// Decodes a stored `!qv` key back into its components (notably `point_id`).
	pub(crate) fn decode_key(k: &[u8]) -> Result<Qv<'_>> {
		Ok(storekey::decode_borrow(k)?)
	}
}

/// Prefix for all `!qv` point payloads of one QORTEX index.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Encode)]
#[storekey(format = "()")]
pub(crate) struct QvRoot<'a> {
	__: u8,
	_a: u8,
	pub ns: NamespaceId,
	_b: u8,
	pub db: DatabaseId,
	_c: u8,
	pub tb: Cow<'a, TableName>,
	_d: u8,
	pub ix: IndexId,
	_e: u8,
	_f: u8,
	_g: u8,
}

impl_kv_key_storekey!(QvRoot<'_> => ());

impl<'a> QvRoot<'a> {
	/// Returns the key range covering every `!qv` payload for one index.
	///
	/// The upper bound increments the final prefix byte (`v` -> `w`) so the range
	/// covers the full `u64` point-id space with no gap.
	pub(crate) fn range(
		ns: NamespaceId,
		db: DatabaseId,
		tb: &'a TableName,
		ix: IndexId,
	) -> Result<Range<Key>> {
		let prefix = Self {
			__: b'/',
			_a: b'*',
			ns,
			_b: b'*',
			db,
			_c: b'*',
			tb: Cow::Borrowed(tb),
			_d: b'+',
			ix,
			_e: b'!',
			_f: b'q',
			_g: b'v',
		}
		.encode_key()?;
		let mut beg = prefix.clone();
		beg.push(0x00);
		let mut end = prefix;
		// Final prefix byte is `v` (0x76); incrementing it yields the exclusive
		// upper bound covering every `!qv{point_id}` key.
		if let Some(last) = end.last_mut() {
			*last += 1;
		}
		Ok(beg..end)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn key() {
		let tb = TableName::from("testtb");
		let val = Qv::new(NamespaceId(1), DatabaseId(2), &tb, IndexId(3), 7);
		let enc = Qv::encode_key(&val).unwrap();
		assert_eq!(
			enc,
			b"/*\x00\x00\x00\x01*\x00\x00\x00\x02*testtb\0+\0\0\0\x03!qv\0\0\0\0\0\0\0\x07"
		);
	}

	#[test]
	fn range_contains_keys() {
		let tb = TableName::from("testtb");
		let range = QvRoot::range(NamespaceId(1), DatabaseId(2), &tb, IndexId(3)).unwrap();
		let k0 = Qv::new(NamespaceId(1), DatabaseId(2), &tb, IndexId(3), 0).encode_key().unwrap();
		let kmax =
			Qv::new(NamespaceId(1), DatabaseId(2), &tb, IndexId(3), u64::MAX).encode_key().unwrap();
		assert!(range.start <= k0 && k0 < range.end);
		assert!(range.start <= kmax && kmax < range.end);
	}
}
