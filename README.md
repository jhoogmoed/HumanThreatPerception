# Human driver risk perception model: Fundamental threat parameters and what makes a situation risky
The level of automation in vehicles is growing. But until all vehicles are completely automated, there will be a transition period where automated vehicles and human drivers coexist. Because these road users coexist, it is necessary that automated vehicles understand human drivers and vice versa, to resolve potential conflicts. This study aims to create a model that predicts human risk perception in different driving scenarios, to provide an understanding of the fundamental features of human threat perception while driving.

The model created is a multi-criteria decision-making process that uses KITTI Vision Benchmark data as an input. This model is tested against the data gathered by an online survey, where 1918 participants answered the question: "How high is the risk on a scale from 0-100?" for 100 situations. Furthermore, a multivariate regression is performed on the survey data, which is compared to the model.

# Dependencies
- python3
- numpy
- pandas
- seaborn
- opencv-python
- sklearn
- matplotlib
