# Dataset Statistics

This table shows the statistics for each dataset in the data directory. These statistics can be used to compare the datasets and understand the class distribution.

## Overview

The following statistics are provided for each dataset:

- **Dataset Name:** The name of the dataset.
- **Number of Instances:** The total number of instances in the dataset.
- **Number of Features:** The total number of features (excluding the target column) in the dataset.
- **Positive Class Frequency (%):** The frequency of the positive class as a percentage of the total instances.
- **Positive Class Count:** The total count of instances in the positive class.
- **Negative Class Count:** The total count of instances in the negative class.

## Example Output

The following is an example output from the `ClassDistribution` class:


| Dataset       | Number of Instances | Number of Features | Positive Class Frequency (%) | Positive Class Count | Negative Class Count |
|---------------|---------------------|--------------------|------------------------------|----------------------|----------------------|
| angina        | 7097                | 140                | 3.64                         | 258                  | 6839                 |
| arthritis     | 7097                | 140                | 42.57                        | 3021                 | 4076                 |
| cataract      | 7097                | 140                | 32.72                        | 2322                 | 4775                 |
| dementia      | 7097                | 140                | 2.09                         | 148                  | 6949                 |
| diabetes      | 7097                | 136                | 13.33                        | 946                  | 6151                 |
| hbp           | 7097                | 140                | 40.21                        | 2854                 | 4243                 |
| heartattack   | 7097                | 140                | 5.65                         | 401                  | 6696                 |
| osteoporosis  | 7097                | 140                | 9.22                         | 654                  | 6443                 |
| parkinsons    | 7097                | 140                | 0.93                         | 66                   | 7031                 |
| stroke        | 7097                | 140                | 5.93                         | 421                  | 6676                 |

## How to Use

1. **Initialization:** Create an instance of the `ClassDistribution` class by providing the path to the directory containing your datasets.
   
   ```python
   from class_distribution import ClassDistribution
   
   data_directory = "./data"
   processor = ClassDistribution(data_directory)
