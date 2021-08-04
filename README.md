# Human driver risk perception model: Fundamental threat parameters and what makes a situation risky
The level of automation in vehicles is growing. But until all vehicles are completely automated, there will be a transition period where automated vehicles and human drivers coexist. Because these road users coexist, it is necessary that automated vehicles understand human drivers and vice versa, to resolve potential conflicts. This study aims to create a model that predicts human risk perception in different driving scenarios, to provide an understanding of the fundamental features of human threat perception while driving.

The model created is a multi-criteria decision-making process that uses KITTI Vision Benchmark data as an input. This model is tested against the data gathered by an online survey, where 1918 participants answered the question: "How high is the risk on a scale from 0-100?" for 100 situations. Furthermore, a multivariate regression is performed on the survey data, which is compared to the model.

# Structure of code

The first step in the pipeline is the parsing of KITTI data from json to txt files. The next step is the selection of images to be used for the survey. With this data, the online survey is started. When the survey finishes, the result have to be parsed again from json and csv files to python dataframes and csv files. These result are then analysyed and compared with the model.

The code is split up into three sections based on the pipeline described above:
1. Parsing of KITTI data and selection of images
2. Parsing of survey result
3. Model creation, optimistation, and comparison


# Dependencies
- python3
- numpy
- pandas
- seaborn
- opencv-python
- sklearn
- matplotlib
