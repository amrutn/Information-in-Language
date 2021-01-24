import pandas as pd

term_data = pd.read_csv('Perceptron Networks/Color/term.txt', sep="\t", header=None)
term_data.columns = ["#Lnum", "#snum", "#cnum", "Term Abbrev"]
cnum_data = pd.read_csv('Perceptron Networks/Color/cnum-vhcm-lab-new.txt', sep="\t")

locations = cnum_data[['#cnum']]
locations['Normalized-L'] = (cnum_data['L*'] - cnum_data['L*'].mean())/(cnum_data['L*'] - cnum_data['L*'].mean()).max()
locations['Normalized-a'] = (cnum_data['a*'] - cnum_data['a*'].mean())/(cnum_data['a*'] - cnum_data['a*'].mean()).std() * 1/2
locations['Normalized-b'] = (cnum_data['b*'] - cnum_data['b*'].mean())/(cnum_data['b*'] - cnum_data['b*'].mean()).std() * 1/2

locations = locations.sort_values('#cnum')
chip_num = list(locations['#cnum'])
lab_norm = [[row[2], row[3], row[4]] for row in locations.itertuples()]

class LanguageData:

    def __init__(self, language_number):
        self.language_number = language_number
    
    def language_data(self):
        isolation = term_data[term_data.get('#Lnum').eq(self.language_number)]
        isolation_grouped = isolation.groupby('#cnum')['Term Abbrev'].apply(list)
        unique_symbols = list(isolation['Term Abbrev'].unique())
        chip_abbrev_percentage = [[(isolation_grouped[i + 1].count(abbrev) / len(isolation_grouped[i + 1])) for abbrev in unique_symbols] for i in range(len(isolation_grouped))]

        result = pd.DataFrame(chip_abbrev_percentage)
        result.index += 1
        result.index.name = '#cnum'
        result.columns = unique_symbols
        return isolation, result
    
    def chip_norm(self):
        percentages = []
        language, distribution = self.language_data()
        for value in chip_num:
            percentages.append((distribution.loc[language["#cnum"] == value]).values.tolist()[0])
        return percentages

    def colors_num(self):
        return len(self.chip_norm()[0])
