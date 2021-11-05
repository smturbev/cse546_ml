## Crime Data Lasso
In this problem you will use provided Coordinate Descent Lasso algorithm and apply it on real-world social data.
Before you dive deep into writing code to generate various plots, you will need to look at the dataset and think through what can go wrong when using it in Machine Learning pipeline.
This is important, because if you will ever work in industry and make decisions about human lives you want to ensure your model is fair and robust.

Start by looking into the file [crime_data_lasso.py](./crime_data_lasso.py).
You will notice that there is only a single `main` function that you need to fill in.
You should not return anything, but create and save all of the plots in the problem.

The reason why we ask you to do so in a single function is because you should **loop through lambdas only once** and log information for **all of the sub-problems at once**.
This way our code will be faster.

To run the problem do: `python homeworks/lasso/crime_data_lasso.py` from the root directory of provided zip file, and then add the plots to your written submission.
