//! Stores the next monotonic point-id counter for a QORTEX index.
//!
//! There is exactly one `!qs` key per index. Its value is the next `u64`
//! point-id to allocate when a new record is indexed. Reusing existing
//! point-ids (for updates) goes through the `!qi` record→point-id mapping.

use std::borrow::Cow;

use storekey::{BorrowDecode, Encode};

use crate::catalog::{DatabaseId, IndexId, NamespaceId};
use crate::kvs::impl_kv_key_storekey;
use crate::val::TableName;

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Encode, BorrowDecode)]
#[storekey(format = "()")]
pub(crate) struct Qs<'a> {
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

impl_kv_key_storekey!(Qs<'_> => u64);

impl<'a> Qs<'a> {
	pub fn new(ns: NamespaceId, db: DatabaseId, tb: &'a TableName, ix: IndexId) -> Self {
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
			_g: b's',
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::kvs::KVKey;

	#[test]
	fn key() {
		let tb = TableName::from("testtb");
		let val = Qs::new(NamespaceId(1), DatabaseId(2), &tb, IndexId(3));
		let enc = Qs::encode_key(&val).unwrap();
		assert_eq!(enc, b"/*\x00\x00\x00\x01*\x00\x00\x00\x02*testtb\0+\0\0\0\x03!qs");
	}
}
