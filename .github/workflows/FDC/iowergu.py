import gpt4free
from gpt4free import Provider, quora, forefront
token = forefront.Account.create(logging=False)
response = gpt4free.Completion.create(
    Provider.ForeFront, prompt='analyze this import numpy as np
from scipy.stats import norm

# Define the probabilities

religion_probabilities = {
    '15-19': {'Hindu': 0.788111076678, 'Muslim': 0.154740578955, 'Christian': 0.020220827616, 'Sikh': 0.018296048182, 'Buddh': 0.006778552025, 'Jain': 0.002982257807, 'Other': 0.006329694187},
    '20-24': {'Hindu': 0.793166510413, 'Muslim': 0.148408796761, 'Christian': 0.020657937692, 'Sikh': 0.018587526376, 'Buddh': 0.007558015461, 'Jain': 0.003399479789, 'Other': 0.005792737316},
    '25-29': {'Hindu': 0.804523735180, 'Muslim': 0.136452372583, 'Christian': 0.021687022614, 'Sikh': 0.017677300622, 'Buddh': 0.007658780130, 'Jain': 0.003789658222, 'Other': 0.006028313846},
    '30-34': {'Hindu': 0.811234241561, 'Muslim': 0.129626480783, 'Christian': 0.022326599908, 'Sikh': 0.017555713557, 'Buddh': 0.007480384197, 'Jain': 0.004120246819, 'Other': 0.005781596579},
    '35-39': {'Hindu': 0.813927582040, 'Muslim': 0.127359106134, 'Christian': 0.022539723954, 'Sikh': 0.017225551319, 'Buddh': 0.007312057926, 'Jain': 0.004026316223, 'Other': 0.005975645362},
    '40-44': {'Hindu': 0.815125645532, 'Muslim': 0.123477649158, 'Christian': 0.023905520641, 'Sikh': 0.018208868594, 'Buddh': 0.007407061949, 'Jain': 0.004375823969, 'Other': 0.006005611449},
    '45-49': {'Hindu': 0.817637618685, 'Muslim': 0.117475281841, 'Christian': 0.025809199631, 'Sikh': 0.019407672771, 'Buddh': 0.007017711120, 'Jain': 0.004909808958, 'Other': 0.006361449835},
    '50-54': {'Hindu': 0.818658717517, 'Muslim': 0.115177392826, 'Christian': 0.026438415330, 'Sikh': 0.019881891089, 'Buddh': 0.007207448161, 'Jain': 0.005321270152, 'Other': 0.005940967368},
    '55-59': {'Hindu': 0.821570473949, 'Muslim': 0.109827132097, 'Christian': 0.028747103980, 'Sikh': 0.019572047961, 'Buddh': 0.007183743513, 'Jain': 0.005780372668, 'Other': 0.005965765235},
    '60-64': {'Hindu': 0.820321778460, 'Muslim': 0.114265248667, 'Christian': 0.025320893784, 'Sikh': 0.021371316661, 'Buddh': 0.006578796454, 'Jain': 0.004998783804, 'Other': 0.005743206157},
    '65-69': {'Hindu': 0.824100613659, 'Muslim': 0.107767990392, 'Christian': 0.024218410445, 'Sikh': 0.024015541636, 'Buddh': 0.007851007461, 'Jain': 0.005163652399, 'Other': 0.005436822280},
    '70-74': {'Hindu': 0.828221916616, 'Muslim': 0.105082122477, 'Christian': 0.023736209267, 'Sikh': 0.023995754442, 'Buddh': 0.007717246824, 'Jain': 0.005146040009, 'Other': 0.004555872616},
    '75-79': {'Hindu': 0.828432172695, 'Muslim': 0.095632368303, 'Christian': 0.030001984143, 'Sikh': 0.025434668796, 'Buddh': 0.007777797325, 'Jain': 0.006950736905, 'Other': 0.004135302096},
    '80+': {'Hindu': 0.816081170469, 'Muslim': 0.107275306391, 'Christian': 0.027013671304, 'Sikh': 0.031298172964, 'Buddh': 0.006766098346, 'Jain': 0.005506373854, 'Other': 0.003860934441}
}

wealth_probabilities = {
    '15-19': {'Poorest': 1, 'Poorer': 0.782894737, 'Middle': 0.559210526, 'Richer': 0.353070175, 'Richest': 0.164473684},
    '20-24': {'Poorest': 1, 'Poorer': 0.8321513, 'Middle': 0.626477541, 'Richer': 0.411347518, 'Richest': 0.196217494},
    '25-29': {'Poorest': 1, 'Poorer': 0.830917874, 'Middle': 0.637681159, 'Richer': 0.43236715, 'Richest': 0.214975845},
    '30-34': {'Poorest': 1, 'Poorer': 0.826446281, 'Middle': 0.641873278, 'Richer': 0.44077135, 'Richest': 0.225895317},
    '35-39': {'Poorest': 1, 'Poorer': 0.817663818, 'Middle': 0.62962963, 'Richer': 0.427350427, 'Richest': 0.213675214},
    '40-44': {'Poorest': 1, 'Poorer': 0.830508475, 'Middle': 0.644067797, 'Richer': 0.440677966, 'Richest': 0.227118644},
    '45-49': {'Poorest': 1, 'Poorer': 0.832236842, 'Middle': 0.644736842, 'Richer': 0.440789474, 'Richest': 0.226973684},
    '50-54': {'Poorest': 1, 'Poorer': 0.835390947, 'Middle': 0.658436214, 'Richer': 0.46090535, 'Richest': 0.251028807},
    '55-59': {'Poorest': 1, 'Poorer': 0.823275862, 'Middle': 0.637931034, 'Richer': 0.443965517, 'Richest': 0.24137931},
    '60-64': {'Poorest': 1, 'Poorer': 0.801843318, 'Middle': 0.608294931, 'Richer': 0.419354839, 'Richest': 0.225806452},
    '65-69': {'Poorest': 1, 'Poorer': 0.807692308, 'Middle': 0.621794872, 'Richer': 0.429487179, 'Richest': 0.230769231},
    '70-74': {'Poorest': 1, 'Poorer': 0.805825243, 'Middle': 0.621359223, 'Richer': 0.427184466, 'Richest': 0.223300971},
    '75-79': {'Poorest': 1, 'Poorer': 0.826923077, 'Middle': 0.634615385, 'Richer': 0.442307692, 'Richest': 0.25},
    '80+': {'Poorest': 1, 'Poorer': 0.813559322, 'Middle': 0.627118644, 'Richer': 0.440677966, 'Richest': 0.237288136}
}

height_mean = 167.6  # cm
height_std = 10.3  # cm

# Get user inputs
age_input = input("Select age group(s) (separated by comma): ").split(',')
religion_input = input("Enter religion: ")
wealth_input = input("Enter minimum wealth status: ")
min_height = float(input("Enter minimum height (in cm): "))

# Calculate the probability

# Probability of age group


# Probability of religion given age group
religion_probability = 1
for age in age_input:
    religion_probability *= religion_probabilities.get(age, {}).get(religion_input, 0)

# Probability of wealth quintile given age group
wealth_probability = 1
for age in age_input:
    wealth_probability *= wealth_probabilities.get(age, {}).get(wealth_input, 0)

# Probability of height
height_probability = 1-(norm.cdf(min_height, loc=height_mean, scale=height_std))

# Calculate the overall probability
overall_probability = religion_probability * wealth_probability * height_probability

# Print the result
print("The probability of fulfilling the criteria is:", overall_probability*100,"%")', model='gpt-4', token=token

)
print(response)
print(f'END')
