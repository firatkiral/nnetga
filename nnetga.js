let Worlds = {
    NumPops: 0,
    Pops: null,
};

const RAND_MAX = 32767;
let randMax = 16;
let randMin = randMax / 2;

function info() {
    console.log("Info:");
    console.log("  Numbers of population:", Worlds.NumPops);
    console.log("");
}

function info_pop() {
    console.log("Population Info:");
    if (Worlds.NumPops === 0) {
        console.log("  No population created");
        return false;
    }

    console.log("  Numbers of population:", Worlds.NumPops);
    for (let i = 0; i < Worlds.NumPops; i++) {
        console.log("  Population index:", Worlds.Pops[iPop].Index);
        console.log("    -number of agents:", Worlds.Pops[iPop].NumAgents);
        console.log("    -number of generation:", Worlds.Pops[iPop].Generation);
        console.log("    -crossover rate:", Worlds.Pops[iPop].CrossRate);
        console.log("    -mutation rate:", Worlds.Pops[iPop].MutateRate);
        console.log("    -mutation max range:", Worlds.Pops[iPop].MutateMax);
        if (Worlds.Pops[iPop].NumAgents > 0) {
            if (Worlds.Pops[iPop].Agents.Net != null) {
                console.log("    -neural network: Yes");
                console.log("      -Numbers of Inputs:", Worlds.Pops[iPop].Agents.Net.NumInputs);
                console.log("      -Numbers of HiddenLayers:", Worlds.Pops[iPop].Agents.Net.NumLayers);
                console.log("      -Numbers of Neurons per Layers:", Worlds.Pops[iPop].Agents.Net.NeuronsPerLayers);
                console.log("      -Numbers of total Neurons:", Worlds.Pops[iPop].Agents.Net.NeuronsPerLayers * Worlds.Pops[iPop].Agents.Net.NumLayers + Worlds.Pops[iPop].Agents.Net.NumOutputs);
                console.log("      -Numbers of Outputs::", Worlds.Pops[iPop].Agents.Net.NumOutputs);
            }
            else {
                console.log("    -neural network: No");
            }
        }
        else {
            console.log("    -neural network: No");
        }
    }
    console.log("");
}

function info_net(iPop, iAgent, level = 0) {
    let iLayer = 0;
    let iNeuron = 0;
    let iWeight = 0;
    let iChromo = 0;
    let iInput = 0;

    let pyChromo = Array(Worlds.Pops[iPop].Agents[iAgent].NumChromo).fill(0);
    for (iChromo = 0; iChromo < Worlds.Pops[iPop].Agents[iAgent].NumChromo; iChromo++) {
        pyChromo[iChromo] = Worlds.Pops[iPop].Agents[iAgent].Chromo[iChromo];
    }

    let pyInput = Array(Worlds.Pops[iPop].Agents[iAgent].Net.NumInputs).fill(0);
    for (iInput = 0; iInput < Worlds.Pops[iPop].Agents[iAgent].Net.NumInputs; iInput++) {
        pyInput[iInput] = Worlds.Pops[iPop].Agents[iAgent].Net.Input[iInput];
    }

    if (level >= 0) {
        console.log("Agent:", Worlds.Pops[iPop].Agents[iAgent].Index, "  NumChromo:", Worlds.Pops[iPop].Agents[iAgent].NumChromo);
    }
    if (level >= 1) {
        console.log(" Chromo:", pyChromo);
        console.log(" Inputs", pyInput);
    }
    if (level >= 2) {
        for (iLayer = 0; iLayer < Worlds.Pops[iPop].Agents[iAgent].Net.NumLayers; iLayer++) {
            console.log("  Layer:", Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Index);
            for (iNeuron = 0; iNeuron < Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].NumNeurons; iNeuron++) {
                console.log("   Neuron:", Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Index);
                for (iWeight = 0; iWeight < Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs; iWeight++) {
                    console.log("    Weights:", Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight]);
                }
            }
        }
    }
}

function add_pop(numPopAdd = 1, crossover = 0.7, mutation = 0.1, mutatemax = 0.3) {
    let iPop = 0;
    let i = 0;

    if (numPopAdd === 0)
        return false;

    Worlds.Pops = [...Array(Worlds.NumPops + numPopAdd)].map(x => { return {}; });

    for (i = 0; i < numPopAdd; i++) {
        iPop = i + Worlds.NumPops;
        Worlds.Pops[iPop].Index = iPop;
        Worlds.Pops[iPop].NumAgents = 0;
        Worlds.Pops[iPop].Agents = null;
        Worlds.Pops[iPop].Generation = 0;
        Worlds.Pops[iPop].CrossRate = crossover;
        Worlds.Pops[iPop].MutateRate = mutation;
        Worlds.Pops[iPop].MutateMax = mutatemax;
    }

    Worlds.NumPops += numPopAdd;

    return true;
}

function add_agent(popIndex = 0, numAgents = 1) {
    let iAgent = 0;

    if (popIndex > (Worlds.NumPops - 1)) {
        console.log(); ('ERROR: popIndex out of range');
        return false;
    }

    Worlds.Pops[popIndex].Agents = [...Array(numAgents)].map(x => { return {}; });
    Worlds.Pops[popIndex].NumAgents = numAgents;

    for (iAgent = 0; iAgent < numAgents; iAgent++) {
        Worlds.Pops[popIndex].Agents[iAgent].Index = iAgent;
        Worlds.Pops[popIndex].Agents[iAgent].Net = null;
        Worlds.Pops[popIndex].Agents[iAgent].NumChromo = 0;
        Worlds.Pops[popIndex].Agents[iAgent].Chromo = null;
        Worlds.Pops[popIndex].Agents[iAgent].Score = 0;
        Worlds.Pops[popIndex].Agents[iAgent].PosLow = 0;
        Worlds.Pops[popIndex].Agents[iAgent].PosHi = 0;
    }

    return true;
}

function add_net(popIndex, num_input, num_layers, num_neurons, num_output) {
    let iAgent = 0;
    let iLayer = 0;
    let iNeuron = 0;
    let iOutput = 0;
    let iWeight = 0;
    let LastLayerIndex = 0;
    let iChromo = 0;
    let iInput = 0;

    if (popIndex > (Worlds.NumPops - 1)) {
        console.log(); ('ERROR: popIndex out of range');
        return false;
    }

    for (iAgent = 0; iAgent < Worlds.Pops[popIndex].NumAgents; iAgent++) {
        iChromo = 0;
        Worlds.Pops[popIndex].Agents[iAgent].Index = iAgent;
        if (num_layers >= 1)
            Worlds.Pops[popIndex].Agents[iAgent].NumChromo = ((num_input + 1) * num_neurons) + ((num_layers - 1) * ((num_neurons + 1) * num_neurons)) + (num_output * (num_neurons + 1));
        else
            Worlds.Pops[popIndex].Agents[iAgent].NumChromo = num_input + 1;

        Worlds.Pops[popIndex].Agents[iAgent].Chromo = Array(Worlds.Pops[popIndex].Agents[iAgent].NumChromo).fill(0.0);
        Worlds.Pops[popIndex].Agents[iAgent].Net = {};
        Worlds.Pops[popIndex].Agents[iAgent].Net.Index = 0;
        Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs = num_input;
        Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers = num_layers + 1;
        Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers = num_neurons;
        Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs = num_output;
        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers = [...Array(Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers)].map(x => { return {}; });

        for (iLayer = 0; iLayer < Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1; iLayer++) {

            Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Index = iLayer;
            Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons = Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers;
            Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons = [...Array(Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers)].map(x => { return {}; });

            for (iNeuron = 0; iNeuron < Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers; iNeuron++) {
                Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Index = iNeuron;
                if (iLayer == 0)
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs + 1;
                else
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers + 1;

                Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Output = 0;

                Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights = Array(Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs).fill(0.0);

                for (iWeight = 0; iWeight < Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs; iWeight++) {
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = clamped_rand(randMin);
                    Worlds.Pops[popIndex].Agents[iAgent].Chromo[iChromo] = Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight];
                    iChromo += 1;
                }
            }
            if (iLayer === Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 2) {
                Worlds.Pops[popIndex].Agents[iAgent].Net.Input = Array(Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs).fill(0.0);
                for (iInput = 0; iInput < Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs; iInput++) {
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Input[iInput] = 333;
                }
                Worlds.Pops[popIndex].Agents[iAgent].Net.Output = [...Array(Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs)].map(x => { return {}; });

                if (Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers >= 2)
                    LastLayerIndex = Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1;
                else
                    LastLayerIndex = 0;

                Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons = Worlds.Pops[popIndex].Agents[iAgent].Net.Output;
                Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Index = Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1;
                Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].NumNeurons = Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs;

                for (iOutput = 0; iOutput < Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs; iOutput++) {
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Index = iOutput;
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Output = 0;
                    if (Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers >= 2)
                        Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers + 1;
                    else
                        Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs + 1;

                    Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Weights = Array(Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs).fill(0.0);

                    for (iWeight = 0; iWeight < Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].NumInputs; iWeight++) {
                        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].Weights[iWeight] = clamped_rand(randMin);
                        Worlds.Pops[popIndex].Agents[iAgent].Chromo[iChromo] = Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].Weights[iWeight];
                        iChromo += 1;
                    }
                }
            }
        }
    }
    return true;
}

function update(Data) {
    let NumData = 0;
    let iData = 0;
    let iPop = 0;
    let iAgent = 0;
    let iLayer = 0;
    let iNeuron = 0;
    let iWeight = 0;
    let sum = 0.0;
    let Bias = -1.0;

    for (iPop = 0; iPop < Worlds.NumPops; iPop++) {
        for (iAgent = 0; iAgent < Worlds.Pops[iPop].NumAgents; iAgent++) {
            NumData = Data[iPop][iAgent].length;
            for (iData = 0; iData < NumData; iData++) {
                Worlds.Pops[iPop].Agents[iAgent].Net.Input[iData] = Data[iPop][iAgent][iData];
            }
        }
    }
    for (iPop = 0; iPop < Worlds.NumPops; iPop++) {
        for (iAgent = 0; iAgent < Worlds.Pops[iPop].NumAgents; iAgent++) {
            for (iLayer = 0; iLayer < Worlds.Pops[iPop].Agents[iAgent].Net.NumLayers; iLayer++) {
                for (iNeuron = 0; iNeuron < Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].NumNeurons; iNeuron++) {
                    sum = 0.0;
                    for (iWeight = 0; iWeight < Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs - 1; iWeight++) {
                        if (iLayer == 0)
                            sum = sum + (Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] * Worlds.Pops[iPop].Agents[iAgent].Net.Input[iWeight]);
                        else {
                            sum = sum + (Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] * Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer - 1].Neurons[iWeight].Output);
                        }
                    }
                    sum = sum + (Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] * Bias);
                    Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Output = sigmoid(sum);
                }
            }
        }
    }


    let ExportData = Array(Worlds.NumPops).fill(1);
    let OutputData = [];
    for (iPop = 0; iPop < Worlds.NumPops; iPop++) {
        ExportData[iPop] = Array(Worlds.Pops[iPop].NumAgents).fill(1);
        for (iAgent = 0; iAgent < Worlds.Pops[iPop].NumAgents; iAgent++) {
            for (let iOutput = 0; iOutput < Worlds.Pops[iPop].Agents[iAgent].Net.NumOutputs; iOutput++) {
                OutputData.push(Worlds.Pops[iPop].Agents[iAgent].Net.Output[iOutput].Output);
            }
            ExportData[iPop][iAgent] = OutputData;
            OutputData = [];
        }
    }
    return ExportData;
}

function insert_chromo(popIndex, agentIndex, Data) {
    let NumData = len(Data);
    let iData = 0;
    let iAgent = 0;
    let iLayer = 0;
    let iNeuron = 0;
    let iWeight = 0;
    let iChromo = 0;

    if (agentIndex === -1) {
        for (iAgent = 0; iAgent < Worlds.Pops[popIndex].NumAgents; iAgent++) {
            iChromo = 0;
            for (iLayer = 0; iLayer < Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers; iLayer++) {
                for (iNeuron = 0; iNeuron < Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons; iNeuron++) {
                    for (iWeight = 0; iWeight < Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs; iWeight++) {
                        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = Data[iChromo];
                        Worlds.Pops[popIndex].Agents[iAgent].Chromo[iChromo] = Data[iChromo];
                        iChromo += 1;
                    }
                }
            }
        }
    }
    else {
        iChromo = 0;
        for (iLayer = 0; iLayer < Worlds.Pops[popIndex].Agents[agentIndex].Net.NumLayers; iLayer++) {
            for (iNeuron = 0; iNeuron < Worlds.Pops[popIndex].Agents[agentIndex].Net.Layers[iLayer].NumNeurons; iNeuron++) {
                for (iWeight = 0; iWeight < Worlds.Pops[popIndex].Agents[agentIndex].Net.Layers[iLayer].Neurons[iNeuron].NumInputs; iWeight++) {
                    Worlds.Pops[popIndex].Agents[agentIndex].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = Data[iChromo];
                    Worlds.Pops[popIndex].Agents[agentIndex].Chromo[iChromo] = Data[iChromo];
                    iChromo += 1;
                }
            }
        }
    }

    return true;
}

function get_chromo(PopIndex, agentIndex) {
    let pyChromo = [];
    for (let iChrome = 0; iChrome < Worlds.Pops[PopIndex].Agents[agentIndex].NumChromo; iChrome++)
        pyChromo.append(Worlds.Pops[PopIndex].Agents[agentIndex].Chromo[iChromo]);

    return pyChromo;
}

function next_gen(PopIndex, Data, crossOption = 1, mutateOption = 1, EliteNum = 0, WorstNum = 0) {
    let TotalScore = 0;
    let PreviousPos = 0;
    let iAgent = 0;
    let iNeuron = 0;
    let iWeight = 0;
    let iChromo = 0;
    let iLovers = 0;
    let iElite = 0;
    let iWorst = 0;
    let Mom = 0;
    let Dad = 0;

    let Baby = [...Array(Worlds.Pops[PopIndex].NumAgents + 1)].map(x => { return {}; });
    let Elite = Array(EliteNum).fill(0);
    let Worst = Array(WorstNum).fill(0);
    let CrossRate = 0;
    let Roulette = 0;
    let CrossPoint = 0;
    let iBaby = 0;
    let MutateChromo = 0;
    let MutateChance = 0;

    for (iBaby = 0; iBaby < Worlds.Pops[PopIndex].NumAgents; iBaby++) {
        Baby[iBaby].Chromo = Array(Worlds.Pops[PopIndex].Agents[0].NumChromo).fill(0.0);
        for (let iChrome = 0; iChrome < Worlds.Pops[PopIndex].Agents[0].NumChromo; iChrome++)
            Baby[iBaby].Chromo[iChromo] = 666.666;
    }
    let RankedData = Data.map((score, index) => ({ score, index })).sort((a, b) => a.score - b.score);

    for (iElite = 0; iElite < EliteNum; iElite++) {
        Elite[iElite] = RankedData[RankedData.length - EliteNum + iElite].index;
    }

    for (iWorst = 0; iWorst < WorstNum; iWorst++) {
        Worst[iWorst] = RankedData[iWorst].index;
        Worlds.Pops[PopIndex].Agents[Worst[iWorst]].Score = 0.0;
    }

    for (iAgent = 0; iAgent < Worlds.Pops[PopIndex].NumAgents; iAgent++) {
        if (arraysearch(iAgent, Worst, WorstNum) === -1)
            Worlds.Pops[PopIndex].Agents[iAgent].Score = Data[iAgent] + 0.0000001;
        TotalScore += Worlds.Pops[PopIndex].Agents[iAgent].Score;
    }

    for (iAgent = 0; iAgent < Worlds.Pops[PopIndex].NumAgents; iAgent++) {
        Worlds.Pops[PopIndex].Agents[iAgent].PosLow = PreviousPos;
        Worlds.Pops[PopIndex].Agents[iAgent].PosHi = (Worlds.Pops[PopIndex].Agents[iAgent].Score / TotalScore) + PreviousPos;
        PreviousPos = Worlds.Pops[PopIndex].Agents[iAgent].PosHi;
    }

    iBaby = 0;
    while (iBaby < (Worlds.Pops[PopIndex].NumAgents)) {
        iLovers = 0;
        while (iLovers < 2) {
            Roulette = (rand() / (RAND_MAX + 1.0));
            for (iAgent = 0; iAgent < Worlds.Pops[PopIndex].NumAgents; iAgent++) {
                if (Roulette > Worlds.Pops[PopIndex].Agents[iAgent].PosLow && Roulette <= Worlds.Pops[PopIndex].Agents[iAgent].PosHi) {
                    if (arraysearch(Worlds.Pops[PopIndex].Agents[iAgent].Index, Worst, WorstNum) === 1)
                        console.log("Hit Worse:", Worlds.Pops[PopIndex].Agents[iAgent].Index);
                    if (iLovers === 0)
                        Mom = Worlds.Pops[PopIndex].Agents[iAgent].Index;
                    if (iLovers === 1)
                        Dad = Worlds.Pops[PopIndex].Agents[iAgent].Index;
                    iLovers += 1;
                }
            }
        }
        CrossRate = (rand() / (RAND_MAX + 1.0));
        CrossPoint = rand() % (Worlds.Pops[PopIndex].Agents[Mom].NumChromo);

        if (crossOption === 0) {
            if (CrossRate <= Worlds.Pops[PopIndex].CrossRate) {
                for (iChromo = 0; iChromo < Worlds.Pops[PopIndex].Agents[Mom].NumChromo; iChromo++) {
                    if (iChromo < CrossPoint) {
                        Baby[iBaby].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo];
                        Baby[iBaby + 1].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo];
                    }
                    else {
                        Baby[iBaby].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo];
                        Baby[iBaby + 1].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo];
                    }
                }
            }
            else {
                for (iChromo = 0; iChromo < Worlds.Pops[PopIndex].Agents[Mom].NumChromo; iChromo++) {
                    Baby[iBaby].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo];
                    Baby[iBaby + 1].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo];
                }
            }
        }
        else if (crossOption >= 1) {
            for (let iChromo = 0; iChromo < Worlds.Pops[PopIndex].Agents[Mom].NumChromo; iChromo++) {
                CrossPoint = (rand() / (RAND_MAX + 1.0));
                if (CrossPoint <= Worlds.Pops[PopIndex].CrossRate) {
                    Baby[iBaby].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo];
                    Baby[iBaby + 1].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo];
                }
                else {
                    Baby[iBaby].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo];
                    Baby[iBaby + 1].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo];
                }
            }
        }
        iBaby += 2;
    }
    iAgent = 0;
    iBaby = 0;
    while (iAgent < (Worlds.Pops[PopIndex].NumAgents)) {
        if (arraysearch(iAgent, Elite, EliteNum) === -1) {
            if (arraysearch(iAgent, Worst, WorstNum) === -1) {
                for (iChromo = 0; iChromo < Worlds.Pops[PopIndex].Agents[iAgent].NumChromo; iChromo++) {
                    Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] = Baby[iBaby].Chromo[iChromo];
                    MutateChance = (rand() / (RAND_MAX + 1.0));
                    if (MutateChance < Worlds.Pops[PopIndex].MutateRate) {
                        if (mutateOption === 0) {
                            MutateChromo = clamped_rand(randMin);
                            Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] = MutateChromo;
                        }
                        else {
                            MutateChromo = clamped_rand(randMin) * Worlds.Pops[PopIndex].MutateMax;
                            Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] += MutateChromo;
                        }
                    }
                }
            }
            else {
                for (iChromo = 0; iChromo < Worlds.Pops[PopIndex].Agents[iAgent].NumChromo; iChromo++)
                    Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] = clamped_rand(randMin);
            }
            iChromo = 0;
            for (let iLayer = 0; iLayer < Worlds.Pops[PopIndex].Agents[iAgent].Net.NumLayers; iLayer++) {
                for (iNeuron = 0; iNeuron < Worlds.Pops[PopIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons; iNeuron++) {
                    for (iWeight = 0; iWeight < Worlds.Pops[PopIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs; iWeight++) {
                        Worlds.Pops[PopIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo];
                        iChromo += 1;
                    }
                }
            }
        }
        Worlds.Pops[PopIndex].Agents[iAgent].PosLow = 0;
        Worlds.Pops[PopIndex].Agents[iAgent].PosHi = 0;
        Worlds.Pops[PopIndex].Agents[iAgent].Score = 0;
        // free(Baby[iAgent].Chromo)
        Baby[iAgent].Chromo.length = 0; // prone to error
        iBaby += 1;
        iAgent += 1;
    }
    // free(Baby) 
    // free(Elite)
    Baby.length = 0; // prone to error
    Elite.length = 0;// prone to error
    Worlds.Pops[PopIndex].Generation += 1;
    return Worlds.Pops[PopIndex].Generation;
}

function rand() {
    return Math.floor(Math.random() * RAND_MAX);
}

function clamped_rand(range) {
    return ((rand() / (RAND_MAX + 1.0)) * range) - ((rand() / (RAND_MAX + 1.0)) * range);
}

function sigmoid(input) {
    return 1 / (1 + (2.7186 ** (-input / 1.0)));
}

function arraysearch(element, array, len) {
    let i = 0;
    for (i = 0; i < len; i++) {
        if (element === array[i])
            return i;
    }
    return -1;
}

// module.exports = {
//     info,
//     info_pop,
//     info_net,
//     add_pop,
//     add_agent,
//     add_net,
//     update,
//     insert_chromo,
//     get_chromo,
//     next_gen
// };

export {
    info,
    info_pop,
    info_net,
    add_pop,
    add_agent,
    add_net,
    update,
    insert_chromo,
    get_chromo,
    next_gen
};
