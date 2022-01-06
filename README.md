# OperationAgnostic

# TODO list

- saving and loading models
- fix visualizing models
- implementing for multiple gym environments
    - cart pole swing up
    - bipedal walking
    - lunar lander
- testing with different operation sets 
    - only uint8 addition 
    - only nand gates 
- fix seeding to work 
- *strech* testing different hyper parameters?
- *strech* testing with different possible depths, can you make wide shallow methods with really low latency for control
- ~make configs in jsons referanceable~
- Cant stop the nand gate or all binary logic gates one from using the sign of the 4th observation value, the pole rotational velocity as a local minima for how to solve it. Literally just a connection from this node to the output gives a reward averaging to ~180 but needs 195 to solve...