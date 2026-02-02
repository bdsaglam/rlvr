# MuSiQue Dataset
It's a multi-hop question answering dataset with 19,938 examples. Each sample has
- question
- answer
- answers
- paragraphs (18-20) including supporting and non-supporting paragraphs
    - 2-4 supporting paragraphs



## Stats

Token count statistics:

Question tokens:
count    19938.000000
mean        19.240897
std          6.047230
min          6.000000
25%         15.000000
50%         18.000000
75%         22.000000
max         57.000000
Name: question, dtype: float64

Answer tokens:
count    19938.000000
mean         3.968252
std          2.414396
min          1.000000
25%          2.000000
50%          3.000000
75%          5.000000
max         25.000000
Name: answer, dtype: float64

All paragraph concatenated tokens:
count    19938.000000
mean      2280.925068
std        491.731783
min        914.000000
25%       1931.000000
50%       2238.000000
75%       2577.750000
max       6029.000000
Name: text, dtype: float64

Supporting paragraph concatenated tokens:
count    19938.000000
mean       288.639583
std        140.539405
min         63.000000
25%        188.000000
50%        258.000000
75%        359.000000
max       1312.000000
Name: supporting_text, dtype: float64