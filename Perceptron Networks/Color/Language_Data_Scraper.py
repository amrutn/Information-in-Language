import pandas as pd

# Filters dataset for specific language for neural network input
# Access to chip percentages and number of colors
class LanguageData:

    def __init__(self, language_number):
        self.language_number = language_number

        # Read and clean data
        # Access to color coordinates
        self.term_data = pd.read_csv('term.txt', sep="\t", header=None)
        self.term_data.columns = ["#Lnum", "#snum", "#cnum", "Term Abbrev"]
        cnum_data = pd.read_csv('cnum-vhcm-lab-new.txt', sep="\t")
        #These normalizations are used to normalize the added noise in automation.py
        self.normalizations = [(cnum_data['L*'] - cnum_data['L*'].mean()).std() * 2,\
         (cnum_data['a*'] - cnum_data['a*'].mean()).std() * 2,\
         (cnum_data['b*'] - cnum_data['b*'].mean()).std() * 2]
        locations = cnum_data[['#cnum']]
        locations['Normalized-L'] = (cnum_data['L*'] - cnum_data['L*'].mean())/self.normalizations[0]
        locations['Normalized-a'] = (cnum_data['a*'] - cnum_data['a*'].mean())/self.normalizations[1]
        locations['Normalized-b'] = (cnum_data['b*'] - cnum_data['b*'].mean())/self.normalizations[2]

        locations = locations.sort_values('#cnum')
        self.chip_num = list(locations['#cnum'])
        self.lab_norm = [[row[2], row[3], row[4]] for row in locations.itertuples()]

    def language_data(self):
        isolation = self.term_data[self.term_data.get('#Lnum').eq(self.language_number)]
        isolation_grouped = isolation.groupby('#cnum')['Term Abbrev'].apply(list)
        unique_symbols = list(isolation['Term Abbrev'].unique())
        chip_abbrev_percentage = [[(isolation_grouped[i + 1].count(abbrev) / len(isolation_grouped[i + 1])) for abbrev in unique_symbols] for i in range(len(isolation_grouped))]

        result = pd.DataFrame(chip_abbrev_percentage)
        result.index += 1
        result.index.name = '#cnum'
        result.columns = unique_symbols
        return result
    
    def chip_norm(self):
        percentages = []
        distribution = self.language_data()
        for value in self.chip_num:
            percentages.append(distribution.loc[value,:].values.tolist())
        return percentages

    def colors_num(self):
        return len(self.chip_norm()[0])
