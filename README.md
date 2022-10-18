# neural_analysis_tools
These are functions/ programmes for analysing various different types of data collected in a systems neuroscience lab

I make use of NEO-core objects for neurophysiological data but functions could also be used without it.
I also make use of click decorators so that functions are callable from the terminal. 
Scripts compatible with Python3.x

Detailed description:

1. Tuning analysis

Contains scripts where the firing of neurons in response to tones of various frequencies and intensities is analyzed to derive the tuning curves for these units

2. Waveform matching 

Contains scripts for determining the similarity between two waveforms (coming from one tetrode each). The method uses a mixture of two gaussians, the parameters of which are learnt using an Expectation-Maximization Algorithm. The initial quantification of the similarity between the waveforms was copied from Tolias et al. 2007.

3. Pupil and Running analyses

Contains scipts anlyze the pupil and running data of the animals and then use the results to further analyze neural data.

4. Behavioural data analysis

Contains scripts for the analysis of home-collected data from auditory Go/No-Go and 2-AFC behavioural experiments. 
