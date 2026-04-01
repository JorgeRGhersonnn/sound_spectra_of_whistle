# Elevator Speech 
## Data Science Student Fellowship 
## Project: Visualization of Sound Spectra of a Whistle 

My name is Jorge Gherson. I'm a junior at Bucknell University, where I am currently studying Physics. 
I hope to answer the question: can we accurately predict the mass flow rate of a system through its sound spectrum? 

Traditionally, measuring flow rate requires expensive, physical flow meters that disrupt the system. Instead, I built an end-to-end machine learning 
pipeline that takes raw signals, breaks it into a spectrum of roughly 1700 frequency bins, and uses those frequencies to calculate the mass flow. 
What makes this topic so interesting is both the physical application and the data science behind it because it offers a completely non-intrusive 
way to measure the flow, with counterintuitive machine learning results. I have been contrasting complex Deep Learning models, like 1D Convolutional Neural Networks, against simpler linear models. Surprisingly, a traditional Ridge Regression model achieved nearly 99% accuracy—vastly outperforming the neural network on our dataset. Right now, I am exploring tree-based models like XGBoost to actually reverse-engineer the physics and isolate the exact frequencies that act as the system's 'secret signature.'