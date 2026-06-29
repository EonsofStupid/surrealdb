/**
[env]
planner-strategy = ["compute-only"]

[test]
reason = "QORTEX (fused Qdrant engine) index for KNN: returns correct nearest neighbours AND the planner routes the query to the qortex index (not the brute-force fallback)."

[[test.results]]
value = "NONE"

[[test.results]]
value = "[{ id: pts:1, point: [1, 2, 3, 4] }]"

[[test.results]]
value = "[{ id: pts:2, point: [4, 5, 6, 7] }]"

[[test.results]]
value = "[{ id: pts:3, point: [8, 9, 10, 11] }]"

[[test.results]]
value = "NONE"

[[test.results]]
value = "[{ id: pts:1 }, { id: pts:2 }]"

[[test.results]]
value = "[{ detail: { plan: { index: 'qortex_pts', operator: '<|2,2|>', value: [2, 3, 4, 5] }, table: 'pts' }, operation: 'Iterate Index' }, { detail: { type: 'Memory' }, operation: 'Collector' }]"
*/

DEFINE INDEX qortex_pts ON pts FIELDS point QORTEX DIMENSION 4 DIST EUCLIDEAN TYPE F32;
CREATE pts:1 SET point = [1,2,3,4];
CREATE pts:2 SET point = [4,5,6,7];
CREATE pts:3 SET point = [8,9,10,11];
LET $pt = [2,3,4,5];
-- nearest 2 to [2,3,4,5] are pts:1 (eucl dist 2) then pts:2 (dist 4)
SELECT id FROM pts WHERE point <|2,EUCLIDEAN|> $pt;
-- EXPLAIN must show Iterate Index on 'qortex_pts' (proves qortex routing, not brute force)
SELECT id FROM pts WHERE point <|2,EUCLIDEAN|> $pt EXPLAIN;
