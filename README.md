# Spark Project on Demographic Analysis of Turkey
This is a project on how to use Spark to handle massive amounts of data by operating on specific data. In python or other supported languages, by using some
operators, we can tell Spark what we wish to do to compute with data, and as a result, it can construct an efficient query plan for execution. In this project, we
not only use low-level RDD API , but also express the same query with high-level DSL operators and the DataFrame API. This report contains experimental analysis, discussion of proposed methodology, errors raised in the process and the corresponding solutions.

The dataset contains some Turkish citizenship information which helps to develop an understanding of the age, sex, and geographical distribution of a population. This leak contains the following information for 49,611,709 Turkish citizens. The leaked database only includes adults of 18 or older and do not includes deceased citizens as of 2009. This leak contains the following information for 49,611,709 Turkish citizens:
1. National Identifier
2. First Name 
3. Last Name 
4. Mother’s First Name 
5. Father’s First Name
6. Gender 
7. City of Birth
8. Date of Birth
9. ID Registration City and District
10. Full Address

Data schema:  
Schema = (uid, national_identifier, first name, last name,mother_first, father_first, gender, birth_city, date_of_birth, id_registration_city, id_registration_district, address_city, address_district, address_neighborhood, street_address, door_or_entrance_number, misc)

Data Sample:   
297107 55711266610 HUSNE GULEC FATMA ALI K KULUNCAK 12/6/1988 MALATYA KULUNCAK MALATYA KULUNCAK SULTANLI KOYU KOYUN KENDISI 13 <NULL>   
297108 55726266100 MENDUH GULEC FATMA MUHUTTIN E KULUNCAK 15/8/1984 MALATYA KULUNCAK MALATYA KULUNCAK SULTANLI KOYU KOYUN KENDISI 79 <NULL>   
297109 55732265982 TEYFIK GULEC RAZIYE SULEYMAN E KULUNCAK 1/1/1984 MALATYA KULUNCAK MALATYA KULUNCAK SULTANLI KOYU KOYUN KENDISI 70 <NULL>

Download here: http://www.sdspeople.fudan.edu.cn/fuyanwei/download/mernis.tar.gz

The source code of E1 -- E6, N1 -- N5 is `/src/part1.py` and the report is  `report1.pdf`.  
E1. Statistics of the oldest male among all citizens of Turkey;   
E2. Count the most common letters in all names;   
E3. The age distribution of the population in the country was counted, and the age groups were (0-18, 19-28, 29-38, 39-48, 49-59, >60);   
E4. Make separate statistics on the number of men and women in the country and calculate the gender ratio;   
E5. Statistics of the months with the highest male and female birth rates in the country;   
E6. Count which street has the largest population.   
N1. The top 10 most common surnames among males and females;   
N2. The average age of citizens in each city is counted, and the aging degree of population in each city is statistically analyzed to judge whether the city is in an aging society (when the population over 60 years old accounts for 10% of the total population in a country or region, or the population over 65 years old accounts for 7% of the total population, It means that the population of this country or region is aging.   
N3. Calculate the two months in which the population in each of the country's top 10 most populous cities have the most concentrated distribution of birthdays; 
N4. Make a statistics of the top 3 surnames in each of the top10 cities with the largest population in the country, and analyze whether the surnames are correlated with the cities in which they are located (correlation analysis can be done by using the data of top10);   
N5. Calculate the two months in which the population of each city has the most concentrated distribution of birthdays among the top 10 cities in the country.   

The source code of N6 -- N7, H1-H4 is `/src/part2.py` and the report is `report2.pdf`.  
N6. Calculate the population density of the top 10 populous cities, where the area of the cities can be searched by Google and the area unit is square kilometers;   N7. According to the origin and residence of the population, the proportion of the trans-administrative floating population and the trans-urban floating population in the total population of Turkey was calculated respectively.   

The data are divided into training set, validation set and test set in 70%, 10% and 20% proportions, and the following questions are discussed in modeling:   
H1. Prediction model for someone's city: Given all information about a person (except the city), predict the city that person is in. The prediction accuracy of the model from Top1 to Top 5 was analyzed.   
H2. Gender prediction model: Based on a given person's information (except gender), predict the gender of that person;   
H3. Name prediction model: Given all information about a person (except the name), predict the most likely last name for that person. The prediction accuracy of the model from Top1 to Top 5 was analyzed.   
H4. Population prediction model: Count the number of births in each year and predict the number of new population in the next year.  

`/src/part1.py`  contains all the code of part 1,  before running it, you need to change the 'file_path' at line 19, where the data file 'data_dump.sql' is stored.

Run the following commands on servers and will get all the results of  E1-N4. 

> spark-submit part1.py

Run the command and you will get results of all the results of N5-H4.

> spark-submit part2.py 

result_part2.ipynb or result_part2.html contains the output which had ever been generated.

