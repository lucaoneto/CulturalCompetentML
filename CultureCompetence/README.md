# Paper Code
This repository contains the code for testing the cultural competence of standard Machine Learning (ML) models.

### How to use the code
Given the Dataset, with the following structure:
- root
- object_type
- culture
- dimension
- color
where, root is the relative path, object_type is "lamps" or "carpets", culture is one of the culture related to the object (lamps-->Chinese, French or Turkish; carpets-->Indian, Japanese or Scandinavian), dimension is the first and second dimension of size of each image, color can be "Greyscale" or "RGB" for example.
One has to enter the paths into the strings file for each dataset.
Note that the given dataset contain images of different sizes, so one has to create another dataset with the desired sizes and colors.

#### Test Standard Machine Learning
Shallow Learning models can be tested only with lamps.
Deep LEarning models can be tested on lamps and carpets.
In order to test Shallow Learning Models (LSVM and GSVM), you need to execute the scripts in "standard" folder.
In order to test Deep Learning Models (RESNET), you need to execute the scripts in "deep_learning" folder.

#### Test Mitigation Strategy
Mitigation strategy can be tested on lamps and carpets.
In order to test Mitigation Strategy (RESNET), you need to exectute the scripts in "deep_learning_mitigation" folder.



