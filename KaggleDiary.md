2026
09
23 This is the first time I participate in competiton. The score was 0.61244. That might be poor, but it was a big step learning how to write basic code.

24 I learned the process of thinking code.(EDA)

25 I copied EDA of sample code.

26 I coded the data visualization part

27-29 I lost the history😭(details 10/03)

30
I had programming classes today, and did homework. I didn't do anything about titanic.

10
01 
I was busy with my presentation homework and did nothing about titanic.

02
I was tired and didn't make progress.

03
Almost all of my history disappeared because I renewed main.py compulsory (git push -f)😭 I made by using vs code.
I consider preprocessing and add something about gender and pclass.


Missing Value Imputation
`Age`, `Fare`: Imputed using median values.
`Embarked`: Imputed using the mode (most frequent value).

Outlier Handling
`Age`: Capped at the 99th percentile (top 1%).

Categorical Encoding
Created a combined feature `Sex_Pclass` (Interaction of Sex and Passenger Class) and applied One-Hot Encoding.
Applied `OrdinalEncoder` to the `Sex` feature.

Data Type Mismatch Resolution
Fixed an evaluation error (scored as 0) caused by `submission.csv` outputting prediction values as floating-point numbers (`float`).
Explicitly cast `PassengerId` and `Survived` to `int` types before exporting to ensure full pipeline reliability.

I struggled with error, so I relied Gemini on them.
My second score was 0.73684
I'll improve it to the nearby 0.8 and next I'll learn about Data Cleaning.

04
I was so unmotivated that I selected only a notebook and read roughly.

05
I struggled with commit on vs code app. It was because I didn't open the titanic folder. I only opened this diary.

06-09
I did'nt make progress anything.

10
I read a high vote notebook and mimiced. I struggled with error. forgot to align the test data format with the training data format.
The third score was 0.75837. I'll change the model not only RandomForest but LightGBM or XGBoost.