# Video Neural Cryptography

This project is aimed to demostrate the capabilities of the Convolutional Neural Networks (ConvNets) in the encryption of sensible data into video sequences. 

## Technical Details

The process of encryption is done through a convolutional neural network that reduces the dimensionality of each frame of the video sequence to be able to introduce the required data in the "flatten image". The encryption is secured through the weights of the ConvNet. To decrypt the data inside the image is needed an ConvNet with the same arquitecture and also the same weights as the ConvNet that encrypted the data, the weigths work as a key. 
## Google Colab 

The code can be runned without the need of installing anything through a public Google Colab notebook.

Google Colab Notebook available here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobinPaspuel/VideoNeuralCryptography/blob/master/NeuralCryptography.ipynb)

> More technical details about the code itself are present in the notebook

Work addapted from the orignial work from Shayan Hashemi available in: [Neural Crytography](https://towardsdatascience.com/neural-cryptography-7733f18184f3)
