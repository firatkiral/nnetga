const RAND_MAX = 32767;

class NNetGA {
    constructor() {
        this.Worlds = {
            NumPops: 0,
            Pops: null,
        };

        this.randMax = 16;
        this.randMin = this.randMax / 2;
    }

    info() {
        console.log("Info:");
        console.log("  Numbers of population:", this.Worlds.NumPops);
        console.log("");
    }

    info_pop() {
        console.log("Population Info:");
        if (this.Worlds.NumPops === 0) {
            console.log("  No population created");
            return false;
        }

        console.log("  Numbers of population:", this.Worlds.NumPops);
        for (let i = 0; i < this.Worlds.NumPops; i++) {
            const pop = this.Worlds.Pops[i];
            console.log("  Population index:", pop.Index);
            console.log("    -number of agents:", pop.NumAgents);
            console.log("    -number of generation:", pop.Generation);
            console.log("    -crossover rate:", pop.CrossRate);
            console.log("    -mutation rate:", pop.MutateRate);
            console.log("    -mutation max range:", pop.MutateMax);
            if (pop.NumAgents > 0) {
                if (pop.Agents && pop.Agents.Net != null) {
                    console.log("    -neural network: Yes");
                    console.log("      -Numbers of Inputs:", pop.Agents.Net.NumInputs);
                    console.log("      -Numbers of HiddenLayers:", pop.Agents.Net.NumLayers);
                    console.log("      -Numbers of Neurons per Layers:", pop.Agents.Net.NeuronsPerLayers);
                    console.log("      -Numbers of total Neurons:", pop.Agents.Net.NeuronsPerLayers * pop.Agents.Net.NumLayers + pop.Agents.Net.NumOutputs);
                    console.log("      -Numbers of Outputs::", pop.Agents.Net.NumOutputs);
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

    info_net(iPop, iAgent, level = 0) {
        let iLayer = 0;
        let iNeuron = 0;
        let iWeight = 0;
        let iChromo = 0;
        let iInput = 0;

        let agent = this.Worlds.Pops[iPop].Agents[iAgent];
        let pyChromo = Array(agent.NumChromo).fill(0);
        for (iChromo = 0; iChromo < agent.NumChromo; iChromo++) {
            pyChromo[iChromo] = agent.Chromo[iChromo];
        }

        let pyInput = Array(agent.Net.NumInputs).fill(0);
        for (iInput = 0; iInput < agent.Net.NumInputs; iInput++) {
            pyInput[iInput] = agent.Net.Input[iInput];
        }

        if (level >= 0) {
            console.log("Agent:", agent.Index, "  NumChromo:", agent.NumChromo);
        }
        if (level >= 1) {
            console.log(" Chromo:", pyChromo);
            console.log(" Inputs", pyInput);
        }
        if (level >= 2) {
            for (iLayer = 0; iLayer < agent.Net.NumLayers; iLayer++) {
                console.log("  Layer:", agent.Net.Layers[iLayer].Index);
                for (iNeuron = 0; iNeuron < agent.Net.Layers[iLayer].NumNeurons; iNeuron++) {
                    console.log("   Neuron:", agent.Net.Layers[iLayer].Neurons[iNeuron].Index);
                    for (iWeight = 0; iWeight < agent.Net.Layers[iLayer].Neurons[iNeuron].NumInputs; iWeight++) {
                        console.log("    Weights:", agent.Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight]);
                    }
                }
            }
        }
    }

    add_pop(numPopAdd = 1, crossover = 0.7, mutation = 0.1, mutatemax = 0.3) {
        let iPop = 0;
        let i = 0;

        if (numPopAdd === 0)
            return false;

        this.Worlds.Pops = [...Array(this.Worlds.NumPops + numPopAdd)].map(x => { return {}; });

        for (i = 0; i < numPopAdd; i++) {
            iPop = i + this.Worlds.NumPops;
            this.Worlds.Pops[iPop].Index = iPop;
            this.Worlds.Pops[iPop].NumAgents = 0;
            this.Worlds.Pops[iPop].Agents = null;
            this.Worlds.Pops[iPop].Generation = 0;
            this.Worlds.Pops[iPop].CrossRate = crossover;
            this.Worlds.Pops[iPop].MutateRate = mutation;
            this.Worlds.Pops[iPop].MutateMax = mutatemax;
        }

        this.Worlds.NumPops += numPopAdd;

        return true;
    }

    add_agent(popIndex = 0, numAgents = 1) {
        let iAgent = 0;

        if (popIndex > (this.Worlds.NumPops - 1)) {
            console.log('ERROR: popIndex out of range');
            return false;
        }

        this.Worlds.Pops[popIndex].Agents = [...Array(numAgents)].map(x => { return {}; });
        this.Worlds.Pops[popIndex].NumAgents = numAgents;

        for (iAgent = 0; iAgent < numAgents; iAgent++) {
            this.Worlds.Pops[popIndex].Agents[iAgent].Index = iAgent;
            this.Worlds.Pops[popIndex].Agents[iAgent].Net = null;
            this.Worlds.Pops[popIndex].Agents[iAgent].NumChromo = 0;
            this.Worlds.Pops[popIndex].Agents[iAgent].Chromo = null;
            this.Worlds.Pops[popIndex].Agents[iAgent].Score = 0;
            this.Worlds.Pops[popIndex].Agents[iAgent].PosLow = 0;
            this.Worlds.Pops[popIndex].Agents[iAgent].PosHi = 0;
        }

        return true;
    }

    add_net(popIndex, num_input, num_layers, num_neurons, num_output) {
        let iAgent = 0;
        let iLayer = 0;
        let iNeuron = 0;
        let iOutput = 0;
        let iWeight = 0;
        let LastLayerIndex = 0;
        let iChromo = 0;
        let iInput = 0;

        if (popIndex > (this.Worlds.NumPops - 1)) {
            console.log('ERROR: popIndex out of range');
            return false;
        }

        for (iAgent = 0; iAgent < this.Worlds.Pops[popIndex].NumAgents; iAgent++) {
            iChromo = 0;
            this.Worlds.Pops[popIndex].Agents[iAgent].Index = iAgent;
            if (num_layers >= 1)
                this.Worlds.Pops[popIndex].Agents[iAgent].NumChromo = ((num_input + 1) * num_neurons) + ((num_layers - 1) * ((num_neurons + 1) * num_neurons)) + (num_output * (num_neurons + 1));
            else
                this.Worlds.Pops[popIndex].Agents[iAgent].NumChromo = num_input + 1;

            this.Worlds.Pops[popIndex].Agents[iAgent].Chromo = Array(this.Worlds.Pops[popIndex].Agents[iAgent].NumChromo).fill(0.0);
            this.Worlds.Pops[popIndex].Agents[iAgent].Net = {};
            this.Worlds.Pops[popIndex].Agents[iAgent].Net.Index = 0;
            this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs = num_input;
            this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers = num_layers + 1;
            this.Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers = num_neurons;
            this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs = num_output;
            this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers = [...Array(this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers)].map(x => { return {}; });

            for (iLayer = 0; iLayer < this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1; iLayer++) {

                this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Index = iLayer;
                this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons = this.Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers;
                this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons = [...Array(this.Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers)].map(x => { return {}; });

                for (iNeuron = 0; iNeuron < this.Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers; iNeuron++) {
                    this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Index = iNeuron;
                    if (iLayer == 0)
                        this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs = this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs + 1;
                    else
                        this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs = this.Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers + 1;

                    this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Output = 0;

                    this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights = Array(this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs).fill(0.0);

                    for (iWeight = 0; iWeight < this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs; iWeight++) {
                        this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = this.clamped_rand(this.randMin);
                        this.Worlds.Pops[popIndex].Agents[iAgent].Chromo[iChromo] = this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight];
                        iChromo += 1;
                    }
                }
                if (iLayer === this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 2) {
                    this.Worlds.Pops[popIndex].Agents[iAgent].Net.Input = Array(this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs).fill(0.0);
                    for (iInput = 0; iInput < this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs; iInput++) {
                        this.Worlds.Pops[popIndex].Agents[iAgent].Net.Input[iInput] = 333;
                    }
                    this.Worlds.Pops[popIndex].Agents[iAgent].Net.Output = [...Array(this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs)].map(x => { return {}; });

                    if (this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers >= 2)
                        LastLayerIndex = this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1;
                    else
                        LastLayerIndex = 0;

                    this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons = this.Worlds.Pops[popIndex].Agents[iAgent].Net.Output;
                    this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Index = this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1;
                    this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].NumNeurons = this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs;

                    for (iOutput = 0; iOutput < this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs; iOutput++) {
                        this.Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Index = iOutput;
                        this.Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Output = 0;
                        if (this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers >= 2)
                            this.Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs = this.Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers + 1;
                        else
                            this.Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs = this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs + 1;

                        this.Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Weights = Array(this.Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs).fill(0.0);

                        for (iWeight = 0; iWeight < this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].NumInputs; iWeight++) {
                            this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].Weights[iWeight] = this.clamped_rand(this.randMin);
                            this.Worlds.Pops[popIndex].Agents[iAgent].Chromo[iChromo] = this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].Weights[iWeight];
                            iChromo += 1;
                        }
                    }
                }
            }
        }
        return true;
    }

    update(Data) {
        let NumData = 0;
        let iData = 0;
        let iPop = 0;
        let iAgent = 0;
        let iLayer = 0;
        let iNeuron = 0;
        let iWeight = 0;
        let sum = 0.0;
        let Bias = -1.0;

        for (iPop = 0; iPop < this.Worlds.NumPops; iPop++) {
            for (iAgent = 0; iAgent < this.Worlds.Pops[iPop].NumAgents; iAgent++) {
                NumData = Data[iPop][iAgent].length;
                for (iData = 0; iData < NumData; iData++) {
                    this.Worlds.Pops[iPop].Agents[iAgent].Net.Input[iData] = Data[iPop][iAgent][iData];
                }
            }
        }
        for (iPop = 0; iPop < this.Worlds.NumPops; iPop++) {
            for (iAgent = 0; iAgent < this.Worlds.Pops[iPop].NumAgents; iAgent++) {
                for (iLayer = 0; iLayer < this.Worlds.Pops[iPop].Agents[iAgent].Net.NumLayers; iLayer++) {
                    for (iNeuron = 0; iNeuron < this.Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].NumNeurons; iNeuron++) {
                        sum = 0.0;
                        for (iWeight = 0; iWeight < this.Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs - 1; iWeight++) {
                            if (iLayer == 0)
                                sum = sum + (this.Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] * this.Worlds.Pops[iPop].Agents[iAgent].Net.Input[iWeight]);
                            else {
                                sum = sum + (this.Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] * this.Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer - 1].Neurons[iWeight].Output);
                            }
                        }
                        sum = sum + (this.Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] * Bias);
                        this.Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Output = this.sigmoid(sum);
                    }
                }
            }
        }


        let ExportData = Array(this.Worlds.NumPops).fill(1);
        let OutputData = [];
        for (iPop = 0; iPop < this.Worlds.NumPops; iPop++) {
            ExportData[iPop] = Array(this.Worlds.Pops[iPop].NumAgents).fill(1);
            for (iAgent = 0; iAgent < this.Worlds.Pops[iPop].NumAgents; iAgent++) {
                for (let iOutput = 0; iOutput < this.Worlds.Pops[iPop].Agents[iAgent].Net.NumOutputs; iOutput++) {
                    OutputData.push(this.Worlds.Pops[iPop].Agents[iAgent].Net.Output[iOutput].Output);
                }
                ExportData[iPop][iAgent] = OutputData;
                OutputData = [];
            }
        }
        return ExportData;
    }

    insert_chromo(popIndex, agentIndex, Data) {
        let iData = 0;
        let iAgent = 0;
        let iLayer = 0;
        let iNeuron = 0;
        let iWeight = 0;
        let iChromo = 0;

        if (agentIndex === -1) {
            for (iAgent = 0; iAgent < this.Worlds.Pops[popIndex].NumAgents; iAgent++) {
                iChromo = 0;
                for (iLayer = 0; iLayer < this.Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers; iLayer++) {
                    for (iNeuron = 0; iNeuron < this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons; iNeuron++) {
                        for (iWeight = 0; iWeight < this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs; iWeight++) {
                            this.Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = Data[iChromo];
                            this.Worlds.Pops[popIndex].Agents[iAgent].Chromo[iChromo] = Data[iChromo];
                            iChromo += 1;
                        }
                    }
                }
            }
        }
        else {
            iChromo = 0;
            for (iLayer = 0; iLayer < this.Worlds.Pops[popIndex].Agents[agentIndex].Net.NumLayers; iLayer++) {
                for (iNeuron = 0; iNeuron < this.Worlds.Pops[popIndex].Agents[agentIndex].Net.Layers[iLayer].NumNeurons; iNeuron++) {
                    for (iWeight = 0; iWeight < this.Worlds.Pops[popIndex].Agents[agentIndex].Net.Layers[iLayer].Neurons[iNeuron].NumInputs; iWeight++) {
                        this.Worlds.Pops[popIndex].Agents[agentIndex].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = Data[iChromo];
                        this.Worlds.Pops[popIndex].Agents[agentIndex].Chromo[iChromo] = Data[iChromo];
                        iChromo += 1;
                    }
                }
            }
        }

        return true;
    }

    get_chromo(PopIndex, agentIndex) {
        let pyChromo = [];
        for (let iChromo = 0; iChromo < this.Worlds.Pops[PopIndex].Agents[agentIndex].NumChromo; iChromo++)
            pyChromo.push(this.Worlds.Pops[PopIndex].Agents[agentIndex].Chromo[iChromo]);

        return pyChromo;
    }

    next_gen(PopIndex, Data, crossOption = 1, mutateOption = 1, EliteNum = 0, WorstNum = 0) {
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

        let Baby = [...Array(this.Worlds.Pops[PopIndex].NumAgents + 1)].map(x => { return {}; });
        let Elite = Array(EliteNum).fill(0);
        let Worst = Array(WorstNum).fill(0);
        let CrossRate = 0;
        let Roulette = 0;
        let CrossPoint = 0;
        let iBaby = 0;
        let MutateChromo = 0;
        let MutateChance = 0;

        for (iBaby = 0; iBaby < this.Worlds.Pops[PopIndex].NumAgents; iBaby++) {
            Baby[iBaby].Chromo = Array(this.Worlds.Pops[PopIndex].Agents[0].NumChromo).fill(0.0);
            for (let iChrome = 0; iChrome < this.Worlds.Pops[PopIndex].Agents[0].NumChromo; iChrome++)
                Baby[iBaby].Chromo[iChrome] = 666.666;
        }
        let RankedData = Data.map((score, index) => ({ score, index })).sort((a, b) => a.score - b.score);

        for (iElite = 0; iElite < EliteNum; iElite++) {
            Elite[iElite] = RankedData[RankedData.length - EliteNum + iElite].index;
        }

        for (iWorst = 0; iWorst < WorstNum; iWorst++) {
            Worst[iWorst] = RankedData[iWorst].index;
            this.Worlds.Pops[PopIndex].Agents[Worst[iWorst]].Score = 0.0;
        }

        for (iAgent = 0; iAgent < this.Worlds.Pops[PopIndex].NumAgents; iAgent++) {
            if (this.arraysearch(iAgent, Worst, WorstNum) === -1)
                this.Worlds.Pops[PopIndex].Agents[iAgent].Score = Data[iAgent] + 0.0000001;
            TotalScore += this.Worlds.Pops[PopIndex].Agents[iAgent].Score;
        }

        for (iAgent = 0; iAgent < this.Worlds.Pops[PopIndex].NumAgents; iAgent++) {
            this.Worlds.Pops[PopIndex].Agents[iAgent].PosLow = PreviousPos;
            this.Worlds.Pops[PopIndex].Agents[iAgent].PosHi = (this.Worlds.Pops[PopIndex].Agents[iAgent].Score / TotalScore) + PreviousPos;
            PreviousPos = this.Worlds.Pops[PopIndex].Agents[iAgent].PosHi;
        }

        iBaby = 0;
        while (iBaby < (this.Worlds.Pops[PopIndex].NumAgents)) {
            iLovers = 0;
            while (iLovers < 2) {
                Roulette = (this.rand() / (RAND_MAX + 1.0));
                for (iAgent = 0; iAgent < this.Worlds.Pops[PopIndex].NumAgents; iAgent++) {
                    if (Roulette > this.Worlds.Pops[PopIndex].Agents[iAgent].PosLow && Roulette <= this.Worlds.Pops[PopIndex].Agents[iAgent].PosHi) {
                        if (this.arraysearch(this.Worlds.Pops[PopIndex].Agents[iAgent].Index, Worst, WorstNum) === 1)
                            console.log("Hit Worse:", this.Worlds.Pops[PopIndex].Agents[iAgent].Index);
                        if (iLovers === 0)
                            Mom = this.Worlds.Pops[PopIndex].Agents[iAgent].Index;
                        if (iLovers === 1)
                            Dad = this.Worlds.Pops[PopIndex].Agents[iAgent].Index;
                        iLovers += 1;
                    }
                }
            }
            CrossRate = (this.rand() / (RAND_MAX + 1.0));
            CrossPoint = this.rand() % (this.Worlds.Pops[PopIndex].Agents[Mom].NumChromo);

            if (crossOption === 0) {
                if (CrossRate <= this.Worlds.Pops[PopIndex].CrossRate) {
                    for (iChromo = 0; iChromo < this.Worlds.Pops[PopIndex].Agents[Mom].NumChromo; iChromo++) {
                        if (iChromo < CrossPoint) {
                            Baby[iBaby].Chromo[iChromo] = this.Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo];
                            Baby[iBaby + 1].Chromo[iChromo] = this.Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo];
                        }
                        else {
                            Baby[iBaby].Chromo[iChromo] = this.Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo];
                            Baby[iBaby + 1].Chromo[iChromo] = this.Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo];
                        }
                    }
                }
                else {
                    for (iChromo = 0; iChromo < this.Worlds.Pops[PopIndex].Agents[Mom].NumChromo; iChromo++) {
                        Baby[iBaby].Chromo[iChromo] = this.Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo];
                        Baby[iBaby + 1].Chromo[iChromo] = this.Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo];
                    }
                }
            }
            else if (crossOption >= 1) {
                for (let iChromo = 0; iChromo < this.Worlds.Pops[PopIndex].Agents[Mom].NumChromo; iChromo++) {
                    CrossPoint = (this.rand() / (RAND_MAX + 1.0));
                    if (CrossPoint <= this.Worlds.Pops[PopIndex].CrossRate) {
                        Baby[iBaby].Chromo[iChromo] = this.Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo];
                        Baby[iBaby + 1].Chromo[iChromo] = this.Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo];
                    }
                    else {
                        Baby[iBaby].Chromo[iChromo] = this.Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo];
                        Baby[iBaby + 1].Chromo[iChromo] = this.Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo];
                    }
                }
            }
            iBaby += 2;
        }
        iAgent = 0;
        iBaby = 0;
        while (iAgent < (this.Worlds.Pops[PopIndex].NumAgents)) {
            if (this.arraysearch(iAgent, Elite, EliteNum) === -1) {
                if (this.arraysearch(iAgent, Worst, WorstNum) === -1) {
                    for (iChromo = 0; iChromo < this.Worlds.Pops[PopIndex].Agents[iAgent].NumChromo; iChromo++) {
                        this.Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] = Baby[iBaby].Chromo[iChromo];
                        MutateChance = (this.rand() / (RAND_MAX + 1.0));
                        if (MutateChance < this.Worlds.Pops[PopIndex].MutateRate) {
                            if (mutateOption === 0) {
                                MutateChromo = this.clamped_rand(this.randMin);
                                this.Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] = MutateChromo;
                            }
                            else {
                                MutateChromo = this.clamped_rand(this.randMin) * this.Worlds.Pops[PopIndex].MutateMax;
                                this.Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] += MutateChromo;
                            }
                        }
                    }
                }
                else {
                    for (iChromo = 0; iChromo < this.Worlds.Pops[PopIndex].Agents[iAgent].NumChromo; iChromo++)
                        this.Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] = this.clamped_rand(this.randMin);
                }
                iChromo = 0;
                for (let iLayer = 0; iLayer < this.Worlds.Pops[PopIndex].Agents[iAgent].Net.NumLayers; iLayer++) {
                    for (iNeuron = 0; iNeuron < this.Worlds.Pops[PopIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons; iNeuron++) {
                        for (iWeight = 0; iWeight < this.Worlds.Pops[PopIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs; iWeight++) {
                            this.Worlds.Pops[PopIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = this.Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo];
                            iChromo += 1;
                        }
                    }
                }
            }
            this.Worlds.Pops[PopIndex].Agents[iAgent].PosLow = 0;
            this.Worlds.Pops[PopIndex].Agents[iAgent].PosHi = 0;
            this.Worlds.Pops[PopIndex].Agents[iAgent].Score = 0;
            // free(Baby[iAgent].Chromo)
            Baby[iAgent].Chromo.length = 0; // prone to error
            iBaby += 1;
            iAgent += 1;
        }
        // free(Baby) 
        // free(Elite)
        Baby.length = 0; // prone to error
        Elite.length = 0;// prone to error
        this.Worlds.Pops[PopIndex].Generation += 1;
        return this.Worlds.Pops[PopIndex].Generation;
    }

    rand() {
        return Math.floor(Math.random() * RAND_MAX);
    }

    clamped_rand(range) {
        return ((this.rand() / (RAND_MAX + 1.0)) * range) - ((this.rand() / (RAND_MAX + 1.0)) * range);
    }

    sigmoid(input) {
        return 1 / (1 + (2.7186 ** (-input / 1.0)));
    }

    arraysearch(element, array, len) {
        let i = 0;
        for (i = 0; i < len; i++) {
            if (element === array[i])
                return i;
        }
        return -1;
    }
}

export { NNetGA };
export default new NNetGA();
