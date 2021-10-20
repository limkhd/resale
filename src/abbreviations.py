abbrev_equivalence_sets = [
    ["AVENUE", "AVE"],
    ["BT", "BUKIT"],
    ["C'WEALTH", "COMMONWEALTH"],
    ["CL", "CLOSE"],
    ["CRES", "CRESCENT"],
    ["CTRL", "CENTRAL"],
    ["DR", "DRIVE"],
    ["GDNS", "GARDENS"],
    ["HTS", "HEIGHTS"],
    ["JLN", "JALAN"],
    ["KG", "KAMPONG"],
    ["LOR", "LORONG"],
    ["MKT", "MARKET"],
    ["NTH", "NORTH"],
    ["PK", "PARK"],
    ["PL", "PLACE"],
    ["RD", "ROAD"],
    ["ST", "STREET"],
    ["STH", "SOUTH"],
    ["TER", "TERRACE"],
    ["TG", "TANJONG"],
    ["UPP", "UPPER"],
]

abbrev_expansion_dict = {}
for equivalence_set in abbrev_equivalence_sets:
    for item in equivalence_set:
        abbrev_expansion_dict[item] = equivalence_set
