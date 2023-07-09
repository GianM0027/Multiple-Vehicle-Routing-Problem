HOW TO RUN THE MODEL:

1) Be sure that a gurobi license (possibly an academic one) is installed in the pc.


2) Open Docker


3) Open the Command Prompt and change directory until you are in the MIP folder.


4) Execute the command: docker build -t mip .


5) Execute the command:

           docker run --env=GRB_CLIENT_LOG=3 --volume=C:\gurobi1002/gurobi.lic:/opt/gurobi/gurobi.lic:ro -it mip

   the script above assumes that the gurobi.lic file is stored in the windows default directory (C:\gurobi1002/gurobi.lic). 
   If the license is stored in another directory replace "C:\gurobi1002/gurobi.lic" with the path where the license is placed.
   The default folders for each OS are:
	
	Windows:    C:\gurobi          or      C:\gurobi1002 (for 10.0.2)
	Linux:	    /opt/gurobi        or      /opt/gurobi1002 (for 10.0.2)
	macOS:	    /Library/gurobi    or      /Library/gurobi1002 (for 10.0.2)


When the program is running it requires two input, an instance number (from 1 to 21) and a configuration number 
(from 1 to 9). The different configurations of the model are described in the report (Chapter 4.4), each number 
refers to a different configuration:

	= 1 -> initialModel
	= 2 -> impliedCons_on_initialModel
	= 3 -> symmBreak_on_initialModel
	= 4 -> impliedAndSymBreak_on_initialModel

	= 5 -> model2
	= 6 -> impliedCons_on_model2
	= 7 -> symmBreak_on_model2
	= 8 -> impliedAndSymBreak_on_model2
	= 9 -> model2_with_focus
