import pandas as pd

data = """
0         ; board_id=0  bucket=1-10  boxes=1  src=1_to_10.sok               manhattan    CPU   SOLVED       0.153647       8             10             40                 15
1         ; board_id=1  bucket=1-10  boxes=1  src=1_to_10.sok               manhattan    CPU   SOLVED       0.028032       1              1              4                  2
2         ; board_id=2  bucket=1-10  boxes=2  src=1_to_10.sok               manhattan    CPU   SOLVED       0.084692       7             12             96                 26
3         ; board_id=3  bucket=1-10  boxes=2  src=1_to_10.sok               manhattan    CPU   SOLVED       0.046613       3              5             40                 18
4         ; board_id=4  bucket=1-10  boxes=2  src=1_to_10.sok               manhattan    CPU   SOLVED       0.090456       6             14            112                 36
5       ; board_id=5  bucket=11-20  boxes=1  src=11_to_20.sok               manhattan    CPU   SOLVED       0.371826      19             34            136                 44
6       ; board_id=6  bucket=11-20  boxes=2  src=11_to_20.sok               manhattan    CPU   SOLVED       0.569238      15            122            976                252
7       ; board_id=7  bucket=11-20  boxes=3  src=11_to_20.sok               manhattan    CPU   SOLVED      24.152300      15            767           9204               1912
8       ; board_id=8  bucket=11-20  boxes=4  src=11_to_20.sok               manhattan    CPU   SOLVED       0.707359      13            153           2448                425
9       ; board_id=9  bucket=11-20  boxes=5  src=11_to_20.sok               manhattan    CPU   SOLVED       1.248830      11            111           2220                678
10     ; board_id=10  bucket=21-30  boxes=1  src=21_to_30.sok               manhattan    CPU   SOLVED       0.307874      22             42            168                 51
11     ; board_id=11  bucket=21-30  boxes=2  src=21_to_30.sok               manhattan    CPU   SOLVED       1.072900      25            263           2104                397
12     ; board_id=12  bucket=21-30  boxes=3  src=21_to_30.sok               manhattan    CPU   SOLVED      24.191900      25           3425          41100               9107
13     ; board_id=13  bucket=21-30  boxes=4  src=21_to_30.sok               manhattan    CPU   SOLVED       2.290370      25            758          12128               1203
14     ; board_id=14  bucket=21-30  boxes=5  src=21_to_30.sok               manhattan    CPU   SOLVED     198.474000      28          31199         623980              76991
15     ; board_id=15  bucket=31-50  boxes=2  src=31_to_40.sok               manhattan    CPU   SOLVED       2.002670      33            369           2952                615
16     ; board_id=16  bucket=31-50  boxes=3  src=31_to_40.sok               manhattan    CPU   SOLVED       4.232750      33           1036          12432               1801
17     ; board_id=17  bucket=31-50  boxes=4  src=31_to_40.sok               manhattan    CPU   SOLVED     528.562000      35          47999         767984             119946
18     ; board_id=18  bucket=31-50  boxes=5  src=41_to_50.sok               manhattan    CPU   SOLVED  107366.000000      42        3946700       78934000           15368034
19     ; board_id=19  bucket=31-50  boxes=6  src=31_to_40.sok               manhattan    CPU   SOLVED    2613.470000      31         432348       10376352            1100195
20     ; board_id=20  bucket=51-70  boxes=3  src=51_to_60.sok               manhattan    CPU   SOLVED     411.218000      52          36776         441312              88620
21     ; board_id=21  bucket=51-70  boxes=4  src=61_to_70.sok               manhattan    CPU   SOLVED    1015.320000      60          77062        1232992             225441
22     ; board_id=22  bucket=51-70  boxes=5  src=61_to_70.sok               manhattan    CPU   SOLVED   43493.700000      64        2497894       49957880            7632708
23     ; board_id=23  bucket=51-70  boxes=6  src=61_to_70.sok               manhattan    CPU   SOLVED    2572.460000      69         406109        9746616             778131
24     ; board_id=24  bucket=51-70  boxes=7  src=61_to_70.sok               manhattan    CPU   SOLVED    2832.430000      62         330255        9247140             805112
25     ; board_id=25  bucket=71-90  boxes=3  src=81_to_90.sok               manhattan    CPU   SOLVED    1425.750000      84          70406         844872             204139
26     ; board_id=26  bucket=71-90  boxes=4  src=81_to_90.sok               manhattan    CPU   SOLVED    1423.820000      87         127596        2041536             306997
27     ; board_id=27  bucket=71-90  boxes=5  src=71_to_80.sok               manhattan    CPU   SOLVED   57221.100000      78        3286072       65721440            9835604
28     ; board_id=28  bucket=71-90  boxes=6  src=71_to_80.sok               manhattan    CPU   SOLVED   16784.800000      77        1838620       44126880            4294874
29     ; board_id=29  bucket=71-90  boxes=7  src=81_to_90.sok               manhattan    CPU  TIMEOUT  120768.000000       0        5156347      144377716           25541145
30   ; board_id=30  bucket=91-100  boxes=7  src=91_to_100.sok               manhattan    CPU  TIMEOUT  120551.000000       0        7559148      211656144           28888841
31   ; board_id=31  bucket=91-100  boxes=7  src=91_to_100.sok               manhattan    CPU  TIMEOUT  120803.000000       0        3521757       98609196           19996121
32   ; board_id=32  bucket=91-100  boxes=6  src=91_to_100.sok               manhattan    CPU  TIMEOUT  120842.000000       0        7422095      178130280           23828223
33   ; board_id=33  bucket=91-100  boxes=7  src=91_to_100.sok               manhattan    CPU  TIMEOUT  120812.000000       0        3698564      103559792           26034667
34   ; board_id=34  bucket=91-100  boxes=7  src=91_to_100.sok               manhattan    CPU  TIMEOUT  121032.000000       0        5542569      155191932           31729821
35      ; board_id=35  bucket=101+  boxes=7  src=101_plus.sok               manhattan    CPU  TIMEOUT  120982.000000       0        4924208      137877824           23454850
36      ; board_id=36  bucket=101+  boxes=6  src=101_plus.sok               manhattan    CPU   SOLVED   28050.200000     100        2861049       68665176            6623317
37      ; board_id=37  bucket=101+  boxes=7  src=101_plus.sok               manhattan    CPU   SOLVED   20519.000000     103        2651104       74230912            5292758
38      ; board_id=38  bucket=101+  boxes=7  src=101_plus.sok               manhattan    CPU   SOLVED   73621.100000     100        7613481      213177468           18723441
39      ; board_id=39  bucket=101+  boxes=7  src=101_plus.sok               manhattan    CPU  TIMEOUT  121078.000000       0        3959193      110857404           24609055
40        ; board_id=0  bucket=1-10  boxes=1  src=1_to_10.sok               hungarian    CPU   SOLVED       0.149217       8              9             36                 15
41        ; board_id=1  bucket=1-10  boxes=1  src=1_to_10.sok               hungarian    CPU   SOLVED       0.020024       1              1              4                  2
42        ; board_id=2  bucket=1-10  boxes=2  src=1_to_10.sok               hungarian    CPU   SOLVED       0.091961       7             12             96                 26
43        ; board_id=3  bucket=1-10  boxes=2  src=1_to_10.sok               hungarian    CPU   SOLVED       0.042394       3              4             32                 16
44        ; board_id=4  bucket=1-10  boxes=2  src=1_to_10.sok               hungarian    CPU   SOLVED       0.077718       6              9             72                 30
45      ; board_id=5  bucket=11-20  boxes=1  src=11_to_20.sok               hungarian    CPU   SOLVED       0.290474      19             23             92                 31
46      ; board_id=6  bucket=11-20  boxes=2  src=11_to_20.sok               hungarian    CPU   SOLVED       0.253091      15             50            400                 96
47      ; board_id=7  bucket=11-20  boxes=3  src=11_to_20.sok               hungarian    CPU   SOLVED       0.848056      15             87           1044                337
48      ; board_id=8  bucket=11-20  boxes=4  src=11_to_20.sok               hungarian    CPU   SOLVED       0.361280      13             58            928                206
49      ; board_id=9  bucket=11-20  boxes=5  src=11_to_20.sok               hungarian    CPU   SOLVED       0.674210      11             53           1060                300
50     ; board_id=10  bucket=21-30  boxes=1  src=21_to_30.sok               hungarian    CPU   SOLVED       0.267438      22             28            112                 43
51     ; board_id=11  bucket=21-30  boxes=2  src=21_to_30.sok               hungarian    CPU   SOLVED       0.887614      25            222           1776                346
52     ; board_id=12  bucket=21-30  boxes=3  src=21_to_30.sok               hungarian    CPU   SOLVED       6.812140      25            998          11976               2847
53     ; board_id=13  bucket=21-30  boxes=4  src=21_to_30.sok               hungarian    CPU   SOLVED       0.700350      25            180           2880                328
54     ; board_id=14  bucket=21-30  boxes=5  src=21_to_30.sok               hungarian    CPU   SOLVED      58.877100      28           8335         166700              22792
55     ; board_id=15  bucket=31-50  boxes=2  src=31_to_40.sok               hungarian    CPU   SOLVED       0.812418      33            149           1192                228
56     ; board_id=16  bucket=31-50  boxes=3  src=31_to_40.sok               hungarian    CPU   SOLVED       3.197500      33            680           8160               1284
57     ; board_id=17  bucket=31-50  boxes=4  src=31_to_40.sok               hungarian    CPU   SOLVED     155.172000      35          12394         198304              39743
58     ; board_id=18  bucket=31-50  boxes=5  src=41_to_50.sok               hungarian    CPU   SOLVED    2648.440000      42         116068        2321360             527548
59     ; board_id=19  bucket=31-50  boxes=6  src=31_to_40.sok               hungarian    CPU   SOLVED      14.267700      31           1999          47976               7973
60     ; board_id=20  bucket=51-70  boxes=3  src=51_to_60.sok               hungarian    CPU   SOLVED     235.821000      52          21001         252012              53854
61     ; board_id=21  bucket=51-70  boxes=4  src=61_to_70.sok               hungarian    CPU   SOLVED     707.894000      60          50671         810736             151729
62     ; board_id=22  bucket=51-70  boxes=5  src=61_to_70.sok               hungarian    CPU   SOLVED     652.072000      64          37709         754180             184054
63     ; board_id=23  bucket=51-70  boxes=6  src=61_to_70.sok               hungarian    CPU   SOLVED    1604.660000      69         236332        5671968             473435
64     ; board_id=24  bucket=51-70  boxes=7  src=61_to_70.sok               hungarian    CPU   SOLVED     187.462000      62          23284         651952              67345
65     ; board_id=25  bucket=71-90  boxes=3  src=81_to_90.sok               hungarian    CPU   SOLVED    1112.100000      84          52379         628548             154153
66     ; board_id=26  bucket=71-90  boxes=4  src=81_to_90.sok               hungarian    CPU   SOLVED    1446.090000      87         121768        1948288             294910
67     ; board_id=27  bucket=71-90  boxes=5  src=71_to_80.sok               hungarian    CPU   SOLVED     633.522000      78          40505         810100             187565
68     ; board_id=28  bucket=71-90  boxes=6  src=71_to_80.sok               hungarian    CPU   SOLVED     392.737000      77          48910        1173840             138253
69     ; board_id=29  bucket=71-90  boxes=7  src=81_to_90.sok               hungarian    CPU   SOLVED     375.734000      81          16597         464716             120543
70   ; board_id=30  bucket=91-100  boxes=7  src=91_to_100.sok               hungarian    CPU   SOLVED     923.933000      97          77989        2183692             372881
71   ; board_id=31  bucket=91-100  boxes=7  src=91_to_100.sok               hungarian    CPU   SOLVED     555.677000      95          23279         651812             136234
72   ; board_id=32  bucket=91-100  boxes=6  src=91_to_100.sok               hungarian    CPU   SOLVED     171.364000      90          15331         367944              57266
73   ; board_id=33  bucket=91-100  boxes=7  src=91_to_100.sok               hungarian    CPU   SOLVED     169.279000      93           8248         230944              56927
74   ; board_id=34  bucket=91-100  boxes=7  src=91_to_100.sok               hungarian    CPU   SOLVED      84.815800      91           4475         125300              37123
75      ; board_id=35  bucket=101+  boxes=7  src=101_plus.sok               hungarian    CPU   SOLVED    3193.010000     102         152140        4259920             865635
76      ; board_id=36  bucket=101+  boxes=6  src=101_plus.sok               hungarian    CPU   SOLVED    1310.310000     100         151548        3637152             433597
77      ; board_id=37  bucket=101+  boxes=7  src=101_plus.sok               hungarian    CPU   SOLVED    2540.130000     103         333837        9347436             744874
78      ; board_id=38  bucket=101+  boxes=7  src=101_plus.sok               hungarian    CPU   SOLVED     166.439000     100          26377         738556              65694
79      ; board_id=39  bucket=101+  boxes=7  src=101_plus.sok               hungarian    CPU   SOLVED     210.735000     105           9294         260232              73037
80        ; board_id=0  bucket=1-10  boxes=1  src=1_to_10.sok       neural_sequential    GPU   SOLVED      64.441900       8              9             36                 15
81        ; board_id=1  bucket=1-10  boxes=1  src=1_to_10.sok       neural_sequential    GPU   SOLVED      11.809900       1              1              4                  2
82        ; board_id=2  bucket=1-10  boxes=2  src=1_to_10.sok       neural_sequential    GPU   SOLVED      87.042700       7             12             96                 26
83        ; board_id=3  bucket=1-10  boxes=2  src=1_to_10.sok       neural_sequential    GPU   SOLVED      63.621600       3              4             32                 16
84        ; board_id=4  bucket=1-10  boxes=2  src=1_to_10.sok       neural_sequential    GPU   SOLVED      53.745200       6              6             48                 22
85      ; board_id=5  bucket=11-20  boxes=1  src=11_to_20.sok       neural_sequential    GPU   SOLVED     116.196000      19             23             92                 31
86      ; board_id=6  bucket=11-20  boxes=2  src=11_to_20.sok       neural_sequential    GPU   SOLVED     511.194000      15            103            824                214
87      ; board_id=7  bucket=11-20  boxes=3  src=11_to_20.sok       neural_sequential    GPU   SOLVED     331.605000      17             57            684                193
88      ; board_id=8  bucket=11-20  boxes=4  src=11_to_20.sok       neural_sequential    GPU   SOLVED     383.942000      13             51            816                200
89      ; board_id=9  bucket=11-20  boxes=5  src=11_to_20.sok       neural_sequential    GPU   SOLVED     168.328000      11             17            340                 97
90     ; board_id=10  bucket=21-30  boxes=1  src=21_to_30.sok       neural_sequential    GPU   SOLVED      75.221300      22             28            112                 43
91     ; board_id=11  bucket=21-30  boxes=2  src=21_to_30.sok       neural_sequential    GPU   SOLVED     688.043000      25            249           1992                371
92     ; board_id=12  bucket=21-30  boxes=3  src=21_to_30.sok       neural_sequential    GPU   SOLVED    8898.520000      25           1533          18396               4245
93     ; board_id=13  bucket=21-30  boxes=4  src=21_to_30.sok       neural_sequential    GPU   SOLVED     296.031000      25             93           1488                142
94     ; board_id=14  bucket=21-30  boxes=5  src=21_to_30.sok       neural_sequential    GPU   SOLVED   27910.100000      30           4520          90400              12667
95     ; board_id=15  bucket=31-50  boxes=2  src=31_to_40.sok       neural_sequential    GPU   SOLVED     681.975000      33            197           1576                319
96     ; board_id=16  bucket=31-50  boxes=3  src=31_to_40.sok       neural_sequential    GPU   SOLVED    1308.430000      33            325           3900                546
97     ; board_id=17  bucket=31-50  boxes=4  src=31_to_40.sok       neural_sequential    GPU   SOLVED   99863.100000      35          12847         205552              42263
98     ; board_id=18  bucket=31-50  boxes=5  src=41_to_50.sok       neural_sequential    GPU  TIMEOUT  120236.000000       0           9636         192720              48545
99     ; board_id=19  bucket=31-50  boxes=6  src=31_to_40.sok       neural_sequential    GPU   SOLVED      15.605300      31           1999          47976               7973
100    ; board_id=20  bucket=51-70  boxes=3  src=51_to_60.sok       neural_sequential    GPU   SOLVED  121635.000000      52          20553         246636              52057
101    ; board_id=21  bucket=51-70  boxes=4  src=61_to_70.sok       neural_sequential    GPU  TIMEOUT  121751.000000       0          16631         266096              49668
102    ; board_id=22  bucket=51-70  boxes=5  src=61_to_70.sok       neural_sequential    GPU  TIMEOUT  125372.000000       0           9136         182720              52479
103    ; board_id=23  bucket=51-70  boxes=6  src=61_to_70.sok       neural_sequential    GPU   SOLVED    1745.600000      69         236332        5671968             473435
104    ; board_id=24  bucket=51-70  boxes=7  src=61_to_70.sok       neural_sequential    GPU   SOLVED     207.108000      62          23284         651952              67345
105    ; board_id=25  bucket=71-90  boxes=3  src=81_to_90.sok       neural_sequential    GPU  TIMEOUT  121930.000000       0          16065         192780              50930
106    ; board_id=26  bucket=71-90  boxes=4  src=81_to_90.sok       neural_sequential    GPU  TIMEOUT  121908.000000       0          18771         300336              54131
107    ; board_id=27  bucket=71-90  boxes=5  src=71_to_80.sok       neural_sequential    GPU  TIMEOUT  122852.000000       0           9570         191400              49154
108    ; board_id=28  bucket=71-90  boxes=6  src=71_to_80.sok       neural_sequential    GPU   SOLVED     428.000000      77          48910        1173840             138253
109    ; board_id=29  bucket=71-90  boxes=7  src=81_to_90.sok       neural_sequential    GPU   SOLVED     418.928000      81          16597         464716             120543
110  ; board_id=30  bucket=91-100  boxes=7  src=91_to_100.sok       neural_sequential    GPU   SOLVED    1024.800000      97          77989        2183692             372881
111  ; board_id=31  bucket=91-100  boxes=7  src=91_to_100.sok       neural_sequential    GPU   SOLVED     612.432000      95          23279         651812             136234
112  ; board_id=32  bucket=91-100  boxes=6  src=91_to_100.sok       neural_sequential    GPU   SOLVED     183.228000      90          15331         367944              57266
113  ; board_id=33  bucket=91-100  boxes=7  src=91_to_100.sok       neural_sequential    GPU   SOLVED     183.111000      93           8248         230944              56927
114  ; board_id=34  bucket=91-100  boxes=7  src=91_to_100.sok       neural_sequential    GPU   SOLVED      97.033800      91           4475         125300              37123
115     ; board_id=35  bucket=101+  boxes=7  src=101_plus.sok       neural_sequential    GPU   SOLVED    3482.700000     102         152140        4259920             865635
116     ; board_id=36  bucket=101+  boxes=6  src=101_plus.sok       neural_sequential    GPU   SOLVED    1413.940000     100         151548        3637152             433597
117     ; board_id=37  bucket=101+  boxes=7  src=101_plus.sok       neural_sequential    GPU   SOLVED    2819.230000     103         333837        9347436             744874
118     ; board_id=38  bucket=101+  boxes=7  src=101_plus.sok       neural_sequential    GPU   SOLVED     189.203000     100          26377         738556              65694
119     ; board_id=39  bucket=101+  boxes=7  src=101_plus.sok       neural_sequential    GPU   SOLVED     234.010000     105           9294         260232              73037
120       ; board_id=0  bucket=1-10  boxes=1  src=1_to_10.sok          neural_batched    GPU   SOLVED     201.213000       8              9             36                 15
121       ; board_id=1  bucket=1-10  boxes=1  src=1_to_10.sok          neural_batched    GPU   SOLVED       9.446140       1              1              4                  2
122       ; board_id=2  bucket=1-10  boxes=2  src=1_to_10.sok          neural_batched    GPU   SOLVED     112.943000       7             12             96                 26
123       ; board_id=3  bucket=1-10  boxes=2  src=1_to_10.sok          neural_batched    GPU   SOLVED      38.046800       3              4             32                 16
124       ; board_id=4  bucket=1-10  boxes=2  src=1_to_10.sok          neural_batched    GPU   SOLVED      55.268300       6              6             48                 22
125     ; board_id=5  bucket=11-20  boxes=1  src=11_to_20.sok          neural_batched    GPU   SOLVED     216.010000      19             23             92                 31
126     ; board_id=6  bucket=11-20  boxes=2  src=11_to_20.sok          neural_batched    GPU   SOLVED     956.401000      15            103            824                214
127     ; board_id=7  bucket=11-20  boxes=3  src=11_to_20.sok          neural_batched    GPU   SOLVED     529.856000      17             57            684                193
128     ; board_id=8  bucket=11-20  boxes=4  src=11_to_20.sok          neural_batched    GPU   SOLVED     474.080000      13             51            816                200
129     ; board_id=9  bucket=11-20  boxes=5  src=11_to_20.sok          neural_batched    GPU   SOLVED     159.734000      11             17            340                 97
130    ; board_id=10  bucket=21-30  boxes=1  src=21_to_30.sok          neural_batched    GPU   SOLVED     262.104000      22             28            112                 43
131    ; board_id=11  bucket=21-30  boxes=2  src=21_to_30.sok          neural_batched    GPU   SOLVED    2052.310000      25            249           1992                371
132    ; board_id=12  bucket=21-30  boxes=3  src=21_to_30.sok          neural_batched    GPU   SOLVED   13984.700000      25           1533          18396               4245
133    ; board_id=13  bucket=21-30  boxes=4  src=21_to_30.sok          neural_batched    GPU   SOLVED     756.803000      25             93           1488                142
134    ; board_id=14  bucket=21-30  boxes=5  src=21_to_30.sok          neural_batched    GPU   SOLVED   38559.100000      30           4520          90400              12667
135    ; board_id=15  bucket=31-50  boxes=2  src=31_to_40.sok          neural_batched    GPU   SOLVED    1739.430000      33            197           1576                319
136    ; board_id=16  bucket=31-50  boxes=3  src=31_to_40.sok          neural_batched    GPU   SOLVED    2575.650000      33            325           3900                546
137    ; board_id=17  bucket=31-50  boxes=4  src=31_to_40.sok          neural_batched    GPU   SOLVED  115914.000000      35          12847         205552              42263
138    ; board_id=18  bucket=31-50  boxes=5  src=41_to_50.sok          neural_batched    GPU  TIMEOUT  124053.000000       0          14135         282700              72092
139    ; board_id=19  bucket=31-50  boxes=6  src=31_to_40.sok          neural_batched    GPU   SOLVED      16.373000      31           1999          47976               7973
140    ; board_id=20  bucket=51-70  boxes=3  src=51_to_60.sok          neural_batched    GPU  TIMEOUT  120937.000000       0          13473         161676              33985
141    ; board_id=21  bucket=51-70  boxes=4  src=61_to_70.sok          neural_batched    GPU  TIMEOUT  122551.000000       0          13899         222384              41494
142    ; board_id=22  bucket=51-70  boxes=5  src=61_to_70.sok          neural_batched    GPU  TIMEOUT  125102.000000       0          13471         269420              77917
143    ; board_id=23  bucket=51-70  boxes=6  src=61_to_70.sok          neural_batched    GPU   SOLVED    1884.550000      69         236332        5671968             473435
144    ; board_id=24  bucket=51-70  boxes=7  src=61_to_70.sok          neural_batched    GPU   SOLVED     215.518000      62          23284         651952              67345
145    ; board_id=25  bucket=71-90  boxes=3  src=81_to_90.sok          neural_batched    GPU  TIMEOUT  123138.000000       0          13597         163164              43652
146    ; board_id=26  bucket=71-90  boxes=4  src=81_to_90.sok          neural_batched    GPU  TIMEOUT  120806.000000       0          13609         217744              39634
147    ; board_id=27  bucket=71-90  boxes=5  src=71_to_80.sok          neural_batched    GPU  TIMEOUT  125338.000000       0          13804         276080              68771
148    ; board_id=28  bucket=71-90  boxes=6  src=71_to_80.sok          neural_batched    GPU   SOLVED     439.931000      77          48910        1173840             138253
149    ; board_id=29  bucket=71-90  boxes=7  src=81_to_90.sok          neural_batched    GPU   SOLVED     420.450000      81          16597         464716             120543
150  ; board_id=30  bucket=91-100  boxes=7  src=91_to_100.sok          neural_batched    GPU   SOLVED    1057.750000      97          77989        2183692             372881
151  ; board_id=31  bucket=91-100  boxes=7  src=91_to_100.sok          neural_batched    GPU   SOLVED     608.192000      95          23279         651812             136234
152  ; board_id=32  bucket=91-100  boxes=6  src=91_to_100.sok          neural_batched    GPU   SOLVED     189.444000      90          15331         367944              57266
153  ; board_id=33  bucket=91-100  boxes=7  src=91_to_100.sok          neural_batched    GPU   SOLVED     187.861000      93           8248         230944              56927
154  ; board_id=34  bucket=91-100  boxes=7  src=91_to_100.sok          neural_batched    GPU   SOLVED      98.000000      91           4475         125300              37123
155     ; board_id=35  bucket=101+  boxes=7  src=101_plus.sok          neural_batched    GPU   SOLVED    3511.570000     102         152140        4259920             865635
156     ; board_id=36  bucket=101+  boxes=6  src=101_plus.sok          neural_batched    GPU   SOLVED    1430.250000     100         151548        3637152             433597
157     ; board_id=37  bucket=101+  boxes=7  src=101_plus.sok          neural_batched    GPU   SOLVED    2852.920000     103         333837        9347436             744874
158     ; board_id=38  bucket=101+  boxes=7  src=101_plus.sok          neural_batched    GPU   SOLVED     193.506000     100          26377         738556              65694
159     ; board_id=39  bucket=101+  boxes=7  src=101_plus.sok          neural_batched    GPU   SOLVED     238.422000     105           9294         260232              73037
160       ; board_id=0  bucket=1-10  boxes=1  src=1_to_10.sok  neural_batched_massive    GPU   SOLVED     181.278000       8             20             80                 23
161       ; board_id=1  bucket=1-10  boxes=1  src=1_to_10.sok  neural_batched_massive    GPU   SOLVED       9.459900       1              1              4                  2
162       ; board_id=2  bucket=1-10  boxes=2  src=1_to_10.sok  neural_batched_massive    GPU   SOLVED      65.588400       7             64            512                150
163       ; board_id=3  bucket=1-10  boxes=2  src=1_to_10.sok  neural_batched_massive    GPU   SOLVED      27.700300       3             15            120                 33
164       ; board_id=4  bucket=1-10  boxes=2  src=1_to_10.sok  neural_batched_massive    GPU   SOLVED      55.701900       6             54            432                 95
165     ; board_id=5  bucket=11-20  boxes=1  src=11_to_20.sok  neural_batched_massive    GPU   SOLVED     177.394000      19             47            188                 51
166     ; board_id=6  bucket=11-20  boxes=2  src=11_to_20.sok  neural_batched_massive    GPU   SOLVED     142.001000      15            330           2640                577
167     ; board_id=7  bucket=11-20  boxes=3  src=11_to_20.sok  neural_batched_massive    GPU   SOLVED     150.039000      15            777           9324               2597
168     ; board_id=8  bucket=11-20  boxes=4  src=11_to_20.sok  neural_batched_massive    GPU   SOLVED     125.216000      13            449           7184               1084
169     ; board_id=9  bucket=11-20  boxes=5  src=11_to_20.sok  neural_batched_massive    GPU   SOLVED     116.233000      11            574          11480               2896
170    ; board_id=10  bucket=21-30  boxes=1  src=21_to_30.sok  neural_batched_massive    GPU   SOLVED     205.605000      22             50            200                 57
171    ; board_id=11  bucket=21-30  boxes=2  src=21_to_30.sok  neural_batched_massive    GPU   SOLVED     235.827000      25            307           2456                456
172    ; board_id=12  bucket=21-30  boxes=3  src=21_to_30.sok  neural_batched_massive    GPU   SOLVED     245.192000      25           1454          17448               4116
173    ; board_id=13  bucket=21-30  boxes=4  src=21_to_30.sok  neural_batched_massive    GPU   SOLVED     239.054000      25           1030          16480               1832
174    ; board_id=14  bucket=21-30  boxes=5  src=21_to_30.sok  neural_batched_massive    GPU   SOLVED    1289.860000      30           8109         162180              22894
175    ; board_id=15  bucket=31-50  boxes=2  src=31_to_40.sok  neural_batched_massive    GPU   SOLVED     314.866000      33            835           6680               1364
176    ; board_id=16  bucket=31-50  boxes=3  src=31_to_40.sok  neural_batched_massive    GPU   SOLVED     317.664000      33           1259          15108               2338
177    ; board_id=17  bucket=31-50  boxes=4  src=31_to_40.sok  neural_batched_massive    GPU   SOLVED    2041.740000      35          12271         196336              41824
178    ; board_id=18  bucket=31-50  boxes=5  src=41_to_50.sok  neural_batched_massive    GPU   SOLVED   28378.600000      42         157876        3157520             771912
179    ; board_id=19  bucket=31-50  boxes=6  src=31_to_40.sok  neural_batched_massive    GPU   SOLVED      44.894200      31           5210         125040              23121
180    ; board_id=20  bucket=51-70  boxes=3  src=51_to_60.sok  neural_batched_massive    GPU   SOLVED    3387.630000      52          20637         247644              52608
181    ; board_id=21  bucket=51-70  boxes=4  src=61_to_70.sok  neural_batched_massive    GPU   SOLVED    4806.800000      60          28793         460688              86418
182    ; board_id=22  bucket=51-70  boxes=5  src=61_to_70.sok  neural_batched_massive    GPU   SOLVED    4098.690000      64          24474         489480             100006
183    ; board_id=23  bucket=51-70  boxes=6  src=61_to_70.sok  neural_batched_massive    GPU   SOLVED    1794.220000      69         243114        5834736             499931
184    ; board_id=24  bucket=51-70  boxes=7  src=61_to_70.sok  neural_batched_massive    GPU   SOLVED     339.810000      62          33497         937916             115038
185    ; board_id=25  bucket=71-90  boxes=3  src=81_to_90.sok  neural_batched_massive    GPU   SOLVED    9078.290000      84          51627         619524             156287
186    ; board_id=26  bucket=71-90  boxes=4  src=81_to_90.sok  neural_batched_massive    GPU   SOLVED   14053.700000      87          85897        1374352             228648
187    ; board_id=27  bucket=71-90  boxes=5  src=71_to_80.sok  neural_batched_massive    GPU   SOLVED   39748.800000      78         237500        4750000             897581
188    ; board_id=28  bucket=71-90  boxes=6  src=71_to_80.sok  neural_batched_massive    GPU   SOLVED     390.737000      77          42278        1014672             133149
189    ; board_id=29  bucket=71-90  boxes=7  src=81_to_90.sok  neural_batched_massive    GPU   SOLVED    3452.990000      81         123268        3451504             976824
190  ; board_id=30  bucket=91-100  boxes=7  src=91_to_100.sok  neural_batched_massive    GPU   SOLVED    1545.520000      97         107762        3017336             634332
191  ; board_id=31  bucket=91-100  boxes=7  src=91_to_100.sok  neural_batched_massive    GPU   SOLVED    2338.890000      95          71752        2009056             552317
192  ; board_id=32  bucket=91-100  boxes=6  src=91_to_100.sok  neural_batched_massive    GPU   SOLVED     550.099000      90          41251         990024             177190
193  ; board_id=33  bucket=91-100  boxes=7  src=91_to_100.sok  neural_batched_massive    GPU   SOLVED    1057.380000      93          47313        1324764             314695
194  ; board_id=34  bucket=91-100  boxes=7  src=91_to_100.sok  neural_batched_massive    GPU   SOLVED     763.962000      91          33387         934836             283354
195     ; board_id=35  bucket=101+  boxes=7  src=101_plus.sok  neural_batched_massive    GPU   SOLVED     984.312000     102          42006        1176168             272609
196     ; board_id=36  bucket=101+  boxes=6  src=101_plus.sok  neural_batched_massive    GPU   SOLVED    1703.470000     100         175024        4200576             531515
197     ; board_id=37  bucket=101+  boxes=7  src=101_plus.sok  neural_batched_massive    GPU   SOLVED    2281.510000     103         262893        7361004             634554
198     ; board_id=38  bucket=101+  boxes=7  src=101_plus.sok  neural_batched_massive    GPU   SOLVED     273.367000     100          35118         983304             109436
199     ; board_id=39  bucket=101+  boxes=7  src=101_plus.sok  neural_batched_massive    GPU   SOLVED    1102.090000     105          44622        1249416             324648
"""

import math

rows = []
for line in data.strip().split('\n'):
    if 'board_id' not in line:
        continue
    parts = line.split()
    board_str = [p for p in parts if 'board_id=' in p][0]
    board_idx = int(board_str.split('=')[1])
    
    device_idx = parts.index('CPU') if 'CPU' in parts else parts.index('GPU')
    heuristic = parts[device_idx - 1]
    
    status = parts[device_idx + 1]
    
    # We want both Nodes and Time!
    # time is at index device_idx + 2
    time_ms = float(parts[device_idx + 2])
    
    # nodes is at index device_idx + 4
    nodes = int(parts[device_idx + 4]) if status in ['SOLVED', 'TIMEOUT', 'OOM'] else 0
    
    # Format strings to show timeouts or OOMs nicely
    def format_val(val, st):
        if st == 'TIMEOUT':
            return f"{val:,.0f} (T)"
        if st == 'OOM':
            return f"{val:,.0f} (OOM)"
        return f"{val:,.0f}"
        
    time_str = format_val(time_ms, status)
    node_str = format_val(nodes, status)
    
    rows.append({
        'board_id': board_idx, 
        'heuristic': heuristic, 
        'nodes': node_str,
        'time': time_str
    })

df = pd.DataFrame(rows)

heuristics = ['manhattan', 'hungarian', 'neural_sequential', 'neural_batched', 'neural_batched_massive']

# Table 1: Nodes
pivot_nodes = df.pivot(index='board_id', columns='heuristic', values='nodes')[heuristics]
print("\n### 🟢 Nodos Expandidos (Los 40 Tableros)\n")
print(pivot_nodes.to_markdown())

# Table 2: Time
pivot_time = df.pivot(index='board_id', columns='heuristic', values='time')[heuristics]
print("\n### ⏱️ Tiempo en Milisegundos (Los 40 Tableros)\n")
print(pivot_time.to_markdown())
