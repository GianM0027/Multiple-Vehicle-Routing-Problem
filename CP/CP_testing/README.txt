HOW TO RUN THE MODEL:

1) Open and Run Docker


2) Open the Command Prompt and change directory until you are in the CP_testing folder.


3) Execute the command: docker build -t 'main_test' .


4) Execute the command:  docker run -ti main_test

When the program is running it requires three integer input, an instance number (from 1 to 21) and a configuration number 
(from 1 to 4) and a sovler number(from 1 to 2). The second objective is not supported using Chuffed.
Configurations:
	= 1 -> Default Model minimizing the 1° objective function
	= 2 -> Default Model minimizing the 2° objective function
	= 3 -> Model with additional constaint minimizing the 1° objective function
	= 4 -> Model with additional constaint minimizing the 2° objective function
Solvers:
	= 1 -> Gecode
	= 2 -> Chuffed
