# Information-in-Language
This is a fun little experiment where we are trying to understand the information carried in language! We attempt to measure the information content of various aspects of language by modeling them with a neural net that is optimized for minimal complexity. For more information, see the document, 'Linguistic Leopards Final Report.pdf' where we explain our approach and results in depth. Our most recent work is in a Google Colab folder that uses some of the files in this repository. 

Our initial toy models came from mathematics. We chose these statements since they are precise and we can easily generate training and validation datasets for them.

To get more meaningful results, we will model color naming conventions in different languages/cultures and determine their complexities. Using our technique, we will attempt to recreate the information theoretic aspects of the paper "Efficient compression in color naming and its evolution" by Zaslavsky, Kemp, Regier and Tishby (2018). We might also try to compare the similarities of different color naming conventions by training a minimal complexity network to recognize color naming conventions in multiple languages simultaneously. The difference between the summed network complexity required to model each language individually and the network complexity required to model both simultaneously can act as a measure of how similar two languages are. 

In addition, we might later model other, more complex, aspects of language such as kinship trees using recurrent neural networks so the network can handle variable length inputs. 

By looking at the trained weights of the networks, this project could help us understand the most efficient underlying structures for neural representations of language. It could also elucidate how growing up around different languages might change the way a person assigns meaning to phrases.

This paper provides a theoretical basis for the complexity of neural networks - http://math.bu.edu/people/mkon/nn30.pdf

This paper analyzes deep neural networks using the Information Bottleneck Principle which we can use to verify our methods - https://arxiv.org/pdf/1503.02406.pdf

Here is a summary on the Information Bottleneck Principle (the wikipedia article is also informative) - https://www.cs.huji.ac.il/labs/learning/Papers/allerton.pdf

Here is a link to the paper on color naming systems - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6077716/
