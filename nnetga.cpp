#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <emscripten/bind.h>

namespace
{

    using DoubleVector = std::vector<double>;
    using AgentInputBatch = std::vector<DoubleVector>;
    using PopulationInputBatch = std::vector<AgentInputBatch>;

    struct Neuron
    {
        int index = 0;
        int num_inputs = 0;
        double output = 0.0;
        DoubleVector weights;
    };

    struct Layer
    {
        int index = 0;
        int num_neurons = 0;
        std::vector<Neuron> neurons;
    };

    struct Net
    {
        int index = 0;
        int num_inputs = 0;
        int num_layers = 0;
        int neurons_per_layer = 0;
        int num_outputs = 0;
        DoubleVector input;
        DoubleVector output;
        std::vector<Layer> layers;
    };

    struct Agent
    {
        int index = 0;
        int num_chromo = 0;
        DoubleVector chromo;
        double score = 0.0;
        double pos_low = 0.0;
        double pos_hi = 0.0;
        Net net;
    };

    struct Population
    {
        int index = 0;
        int num_agents = 0;
        int generation = 0;
        double cross_rate = 0.7;
        double mutate_rate = 0.1;
        double mutate_max = 0.3;
        std::vector<Agent> agents;
    };

    class World
    {
    public:
        int num_pops = 0;
        std::vector<Population> pops;

        bool add_pop(int num_pop_add = 1, double crossover = 0.7, double mutation = 0.1, double mutatemax = 0.3)
        {
            if (num_pop_add <= 0)
            {
                return false;
            }

            const int start_index = num_pops;
            pops.resize(static_cast<std::size_t>(num_pops + num_pop_add));
            for (int offset = 0; offset < num_pop_add; ++offset)
            {
                Population &pop = pops[static_cast<std::size_t>(start_index + offset)];
                pop.index = start_index + offset;
                pop.num_agents = 0;
                pop.generation = 0;
                pop.cross_rate = crossover;
                pop.mutate_rate = mutation;
                pop.mutate_max = mutatemax;
                pop.agents.clear();
            }

            num_pops += num_pop_add;
            return true;
        }

        bool add_agent(int pop_index = 0, int num_agents = 1)
        {
            if (!valid_pop(pop_index) || num_agents <= 0)
            {
                return false;
            }

            Population &pop = pops[static_cast<std::size_t>(pop_index)];
            pop.num_agents = num_agents;
            pop.agents.assign(static_cast<std::size_t>(num_agents), Agent{});

            for (int agent_index = 0; agent_index < num_agents; ++agent_index)
            {
                Agent &agent = pop.agents[static_cast<std::size_t>(agent_index)];
                agent.index = agent_index;
                agent.num_chromo = 0;
                agent.score = 0.0;
                agent.pos_low = 0.0;
                agent.pos_hi = 0.0;
            }

            return true;
        }

        bool add_net(int pop_index, int num_input, int num_layers, int num_neurons, int num_output)
        {
            if (!valid_pop(pop_index))
            {
                return false;
            }

            Population &pop = pops[static_cast<std::size_t>(pop_index)];
            if (pop.num_agents <= 0)
            {
                return false;
            }

            for (int agent_index = 0; agent_index < pop.num_agents; ++agent_index)
            {
                Agent &agent = pop.agents[static_cast<std::size_t>(agent_index)];
                configure_agent_net(agent, num_input, num_layers, num_neurons, num_output);
                initialize_agent_weights(agent);
            }

            return true;
        }

        PopulationInputBatch update(const PopulationInputBatch &data)
        {
            for (int pop_index = 0; pop_index < num_pops; ++pop_index)
            {
                if (pop_index >= static_cast<int>(data.size()))
                {
                    break;
                }

                Population &pop = pops[static_cast<std::size_t>(pop_index)];
                for (int agent_index = 0; agent_index < pop.num_agents; ++agent_index)
                {
                    if (agent_index >= static_cast<int>(data[static_cast<std::size_t>(pop_index)].size()))
                    {
                        break;
                    }

                    Agent &agent = pop.agents[static_cast<std::size_t>(agent_index)];
                    const DoubleVector &input_values = data[static_cast<std::size_t>(pop_index)][static_cast<std::size_t>(agent_index)];
                    const int input_count = std::min(static_cast<int>(input_values.size()), agent.net.num_inputs);
                    for (int input_index = 0; input_index < input_count; ++input_index)
                    {
                        agent.net.input[static_cast<std::size_t>(input_index)] = input_values[static_cast<std::size_t>(input_index)];
                    }
                }
            }

            for (int pop_index = 0; pop_index < num_pops; ++pop_index)
            {
                Population &pop = pops[static_cast<std::size_t>(pop_index)];
                for (int agent_index = 0; agent_index < pop.num_agents; ++agent_index)
                {
                    run_forward_pass(pop.agents[static_cast<std::size_t>(agent_index)]);
                }
            }

            PopulationInputBatch output(static_cast<std::size_t>(num_pops));
            for (int pop_index = 0; pop_index < num_pops; ++pop_index)
            {
                Population &pop = pops[static_cast<std::size_t>(pop_index)];
                output[static_cast<std::size_t>(pop_index)].resize(static_cast<std::size_t>(pop.num_agents));

                for (int agent_index = 0; agent_index < pop.num_agents; ++agent_index)
                {
                    output[static_cast<std::size_t>(pop_index)][static_cast<std::size_t>(agent_index)] =
                        pop.agents[static_cast<std::size_t>(agent_index)].net.output;
                }
            }

            return output;
        }

        bool insert_chromo(int pop_index, int agent_index, const DoubleVector &data)
        {
            if (!valid_pop(pop_index))
            {
                return false;
            }

            Population &pop = pops[static_cast<std::size_t>(pop_index)];
            if (agent_index == -1)
            {
                for (int current_agent = 0; current_agent < pop.num_agents; ++current_agent)
                {
                    apply_chromo_to_agent(pop.agents[static_cast<std::size_t>(current_agent)], data);
                }
                return true;
            }

            if (agent_index < 0 || agent_index >= pop.num_agents)
            {
                return false;
            }

            apply_chromo_to_agent(pop.agents[static_cast<std::size_t>(agent_index)], data);
            return true;
        }

        DoubleVector get_chromo(int pop_index, int agent_index) const
        {
            if (!valid_pop(pop_index))
            {
                return {};
            }

            const Population &pop = pops[static_cast<std::size_t>(pop_index)];
            if (agent_index < 0 || agent_index >= pop.num_agents)
            {
                return {};
            }

            return pop.agents[static_cast<std::size_t>(agent_index)].chromo;
        }

        int next_gen(int pop_index, const DoubleVector &data, int cross_option = 1, int mutate_option = 1, int elite_num = 0, int worst_num = 0)
        {
            if (!valid_pop(pop_index))
            {
                return -1;
            }

            Population &pop = pops[static_cast<std::size_t>(pop_index)];
            if (pop.num_agents <= 0)
            {
                return pop.generation;
            }

            const int agent_count = pop.num_agents;
            elite_num = clamp_count(elite_num, agent_count);
            worst_num = clamp_count(worst_num, agent_count - elite_num);

            std::vector<int> ranked_indices(static_cast<std::size_t>(agent_count));
            std::iota(ranked_indices.begin(), ranked_indices.end(), 0);
            std::sort(ranked_indices.begin(), ranked_indices.end(), [&](int left, int right)
                      {
            const double left_score = left < static_cast<int>(data.size()) ? data[static_cast<std::size_t>(left)] : 0.0;
            const double right_score = right < static_cast<int>(data.size()) ? data[static_cast<std::size_t>(right)] : 0.0;
            return left_score < right_score; });

            std::vector<bool> elite_mask(static_cast<std::size_t>(agent_count), false);
            std::vector<bool> worst_mask(static_cast<std::size_t>(agent_count), false);

            for (int i = 0; i < elite_num; ++i)
            {
                elite_mask[static_cast<std::size_t>(ranked_indices[static_cast<std::size_t>(agent_count - elite_num + i)])] = true;
            }
            for (int i = 0; i < worst_num; ++i)
            {
                worst_mask[static_cast<std::size_t>(ranked_indices[static_cast<std::size_t>(i)])] = true;
            }

            double total_score = 0.0;
            for (int agent_index = 0; agent_index < agent_count; ++agent_index)
            {
                Agent &agent = pop.agents[static_cast<std::size_t>(agent_index)];
                if (worst_mask[static_cast<std::size_t>(agent_index)])
                {
                    agent.score = 0.0;
                }
                else
                {
                    const double base_score = agent_index < static_cast<int>(data.size()) ? data[static_cast<std::size_t>(agent_index)] : 0.0;
                    agent.score = base_score + 0.0000001;
                }
                total_score += agent.score;
            }

            double previous_pos = 0.0;
            if (total_score <= std::numeric_limits<double>::epsilon())
            {
                const double step = 1.0 / static_cast<double>(agent_count);
                for (int agent_index = 0; agent_index < agent_count; ++agent_index)
                {
                    Agent &agent = pop.agents[static_cast<std::size_t>(agent_index)];
                    agent.pos_low = previous_pos;
                    agent.pos_hi = previous_pos + step;
                    previous_pos = agent.pos_hi;
                }
            }
            else
            {
                for (int agent_index = 0; agent_index < agent_count; ++agent_index)
                {
                    Agent &agent = pop.agents[static_cast<std::size_t>(agent_index)];
                    agent.pos_low = previous_pos;
                    agent.pos_hi = previous_pos + (agent.score / total_score);
                    previous_pos = agent.pos_hi;
                }
            }

            for (int agent_index = 0; agent_index < agent_count; ++agent_index)
            {
                Agent &agent = pop.agents[static_cast<std::size_t>(agent_index)];
                if (elite_mask[static_cast<std::size_t>(agent_index)])
                {
                    continue;
                }

                if (worst_mask[static_cast<std::size_t>(agent_index)])
                {
                    randomize_chromo(agent.chromo, agent.num_chromo);
                    sync_chromo_to_net(agent);
                    continue;
                }

                const int mom_index = select_parent(pop);
                const int dad_index = select_parent(pop);
                const Agent &mom = pop.agents[static_cast<std::size_t>(mom_index)];
                const Agent &dad = pop.agents[static_cast<std::size_t>(dad_index)];

                DoubleVector baby = breed_chromo(mom.chromo, dad.chromo, pop.cross_rate, cross_option);
                mutate_chromo(baby, pop.mutate_rate, pop.mutate_max, mutate_option);
                agent.chromo = std::move(baby);
                sync_chromo_to_net(agent);
            }

            for (int agent_index = 0; agent_index < agent_count; ++agent_index)
            {
                Agent &agent = pop.agents[static_cast<std::size_t>(agent_index)];
                agent.pos_low = 0.0;
                agent.pos_hi = 0.0;
                agent.score = 0.0;
            }

            pop.generation += 1;
            return pop.generation;
        }

        void info() const
        {
            std::cout << "Info:\n";
            std::cout << "  Numbers of population: " << num_pops << "\n\n";
        }

        void info_pop() const
        {
            std::cout << "Population Info:\n";
            if (num_pops == 0)
            {
                std::cout << "  No population created\n";
                return;
            }

            std::cout << "  Numbers of population: " << num_pops << "\n";
            for (int pop_index = 0; pop_index < num_pops; ++pop_index)
            {
                const Population &pop = pops[static_cast<std::size_t>(pop_index)];
                std::cout << "  Population index: " << pop.index << "\n";
                std::cout << "    -number of agents: " << pop.num_agents << "\n";
                std::cout << "    -number of generation: " << pop.generation << "\n";
                std::cout << "    -crossover rate: " << pop.cross_rate << "\n";
                std::cout << "    -mutation rate: " << pop.mutate_rate << "\n";
                std::cout << "    -mutation max range: " << pop.mutate_max << "\n";

                if (pop.num_agents > 0 && !pop.agents.empty() && !pop.agents.front().net.layers.empty())
                {
                    const Net &net = pop.agents.front().net;
                    std::cout << "    -neural network: Yes\n";
                    std::cout << "      -Numbers of Inputs: " << net.num_inputs << "\n";
                    std::cout << "      -Numbers of HiddenLayers: " << net.num_layers << "\n";
                    std::cout << "      -Numbers of Neurons per Layers: " << net.neurons_per_layer << "\n";
                    std::cout << "      -Numbers of total Neurons: "
                              << (net.neurons_per_layer * net.num_layers + net.num_outputs) << "\n";
                    std::cout << "      -Numbers of Outputs: " << net.num_outputs << "\n";
                }
                else
                {
                    std::cout << "    -neural network: No\n";
                }
            }
            std::cout << "\n";
        }

        void info_net(int pop_index, int agent_index, int level = 0) const
        {
            if (!valid_pop(pop_index))
            {
                return;
            }

            const Population &pop = pops[static_cast<std::size_t>(pop_index)];
            if (agent_index < 0 || agent_index >= pop.num_agents)
            {
                return;
            }

            const Agent &agent = pop.agents[static_cast<std::size_t>(agent_index)];
            if (level >= 0)
            {
                std::cout << "Agent: " << agent.index << "  NumChromo: " << agent.num_chromo << "\n";
            }
            if (level >= 1)
            {
                std::cout << " Chromo: [";
                for (std::size_t i = 0; i < agent.chromo.size(); ++i)
                {
                    if (i > 0)
                    {
                        std::cout << ", ";
                    }
                    std::cout << agent.chromo[i];
                }
                std::cout << "]\n";

                std::cout << " Inputs: [";
                for (std::size_t i = 0; i < agent.net.input.size(); ++i)
                {
                    if (i > 0)
                    {
                        std::cout << ", ";
                    }
                    std::cout << agent.net.input[i];
                }
                std::cout << "]\n";
            }
            if (level >= 2)
            {
                for (const Layer &layer : agent.net.layers)
                {
                    std::cout << "  Layer: " << layer.index << "\n";
                    for (const Neuron &neuron : layer.neurons)
                    {
                        std::cout << "   Neuron: " << neuron.index << "\n";
                        for (double weight : neuron.weights)
                        {
                            std::cout << "    Weights: " << weight << "\n";
                        }
                    }
                }
            }
        }

    private:
        static double rand_unit()
        {
            static thread_local std::mt19937 generator{std::random_device{}()};
            static thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
            return distribution(generator);
        }

        static double clamped_rand(double range)
        {
            return (rand_unit() * 2.0 - 1.0) * range;
        }

        static double sigmoid(double input)
        {
            return 1.0 / (1.0 + std::exp(-input));
        }

        static int clamp_count(int value, int limit)
        {
            if (value < 0)
            {
                return 0;
            }
            return std::min(value, limit);
        }

        bool valid_pop(int pop_index) const
        {
            return pop_index >= 0 && pop_index < num_pops;
        }

        static int chromosome_count(int num_input, int num_layers, int num_neurons, int num_output)
        {
            if (num_layers <= 0)
            {
                return (num_input + 1) * num_output;
            }

            return ((num_input + 1) * num_neurons) + ((num_layers - 1) * ((num_neurons + 1) * num_neurons)) + ((num_neurons + 1) * num_output);
        }

        static void configure_layer(Layer &layer, int layer_index, int neuron_count, int num_inputs_per_neuron, double range)
        {
            layer.index = layer_index;
            layer.num_neurons = neuron_count;
            layer.neurons.assign(static_cast<std::size_t>(neuron_count), Neuron{});

            for (int neuron_index = 0; neuron_index < neuron_count; ++neuron_index)
            {
                Neuron &neuron = layer.neurons[static_cast<std::size_t>(neuron_index)];
                neuron.index = neuron_index;
                neuron.num_inputs = num_inputs_per_neuron;
                neuron.output = 0.0;
                neuron.weights.assign(static_cast<std::size_t>(num_inputs_per_neuron), 0.0);
                for (double &weight : neuron.weights)
                {
                    weight = clamped_rand(range);
                }
            }
        }

        void configure_agent_net(Agent &agent, int num_input, int num_layers, int num_neurons, int num_output)
        {
            agent.num_chromo = chromosome_count(num_input, num_layers, num_neurons, num_output);
            agent.chromo.assign(static_cast<std::size_t>(agent.num_chromo), 0.0);

            agent.net.index = 0;
            agent.net.num_inputs = num_input;
            agent.net.num_layers = std::max(0, num_layers) + 1;
            agent.net.neurons_per_layer = num_neurons;
            agent.net.num_outputs = num_output;
            agent.net.input.assign(static_cast<std::size_t>(num_input), 0.0);
            agent.net.output.assign(static_cast<std::size_t>(num_output), 0.0);
            agent.net.layers.clear();
            agent.net.layers.resize(static_cast<std::size_t>(agent.net.num_layers));
        }

        void initialize_agent_weights(Agent &agent)
        {
            int chromo_index = 0;
            const int hidden_layer_count = agent.net.num_layers - 1;

            if (hidden_layer_count <= 0)
            {
                Layer &output_layer = agent.net.layers.front();
                configure_layer(output_layer, 0, agent.net.num_outputs, agent.net.num_inputs + 1, 16.0);
                for (Neuron &neuron : output_layer.neurons)
                {
                    for (double &weight : neuron.weights)
                    {
                        agent.chromo[static_cast<std::size_t>(chromo_index++)] = weight;
                    }
                }
                return;
            }

            for (int layer_index = 0; layer_index < hidden_layer_count; ++layer_index)
            {
                const int inputs_per_neuron = (layer_index == 0) ? (agent.net.num_inputs + 1) : (agent.net.neurons_per_layer + 1);
                configure_layer(agent.net.layers[static_cast<std::size_t>(layer_index)], layer_index, agent.net.neurons_per_layer, inputs_per_neuron, 16.0);
                for (Neuron &neuron : agent.net.layers[static_cast<std::size_t>(layer_index)].neurons)
                {
                    for (double &weight : neuron.weights)
                    {
                        agent.chromo[static_cast<std::size_t>(chromo_index++)] = weight;
                    }
                }
            }

            Layer &output_layer = agent.net.layers.back();
            configure_layer(output_layer, hidden_layer_count, agent.net.num_outputs, agent.net.neurons_per_layer + 1, 16.0);
            for (Neuron &neuron : output_layer.neurons)
            {
                for (double &weight : neuron.weights)
                {
                    agent.chromo[static_cast<std::size_t>(chromo_index++)] = weight;
                }
            }
        }

        void run_forward_pass(Agent &agent)
        {
            if (agent.net.layers.empty())
            {
                return;
            }

            constexpr double bias = -1.0;
            const int hidden_layer_count = agent.net.num_layers - 1;

            if (hidden_layer_count <= 0)
            {
                Layer &output_layer = agent.net.layers.front();
                for (Neuron &neuron : output_layer.neurons)
                {
                    double sum = 0.0;
                    for (int input_index = 0; input_index < agent.net.num_inputs; ++input_index)
                    {
                        sum += neuron.weights[static_cast<std::size_t>(input_index)] * agent.net.input[static_cast<std::size_t>(input_index)];
                    }
                    sum += neuron.weights.back() * bias;
                    neuron.output = sigmoid(sum);
                }

                agent.net.output.resize(static_cast<std::size_t>(agent.net.num_outputs));
                for (int output_index = 0; output_index < agent.net.num_outputs; ++output_index)
                {
                    agent.net.output[static_cast<std::size_t>(output_index)] = output_layer.neurons[static_cast<std::size_t>(output_index)].output;
                }
                return;
            }

            for (int layer_index = 0; layer_index < hidden_layer_count; ++layer_index)
            {
                Layer &layer = agent.net.layers[static_cast<std::size_t>(layer_index)];
                const bool first_layer = layer_index == 0;
                for (Neuron &neuron : layer.neurons)
                {
                    double sum = 0.0;
                    if (first_layer)
                    {
                        for (int input_index = 0; input_index < agent.net.num_inputs; ++input_index)
                        {
                            sum += neuron.weights[static_cast<std::size_t>(input_index)] * agent.net.input[static_cast<std::size_t>(input_index)];
                        }
                    }
                    else
                    {
                        const Layer &previous_layer = agent.net.layers[static_cast<std::size_t>(layer_index - 1)];
                        for (int prev_index = 0; prev_index < previous_layer.num_neurons; ++prev_index)
                        {
                            sum += neuron.weights[static_cast<std::size_t>(prev_index)] * previous_layer.neurons[static_cast<std::size_t>(prev_index)].output;
                        }
                    }

                    sum += neuron.weights.back() * bias;
                    neuron.output = sigmoid(sum);
                }
            }

            Layer &output_layer = agent.net.layers.back();
            const Layer &previous_layer = agent.net.layers[static_cast<std::size_t>(hidden_layer_count - 1)];
            for (Neuron &neuron : output_layer.neurons)
            {
                double sum = 0.0;
                for (int prev_index = 0; prev_index < previous_layer.num_neurons; ++prev_index)
                {
                    sum += neuron.weights[static_cast<std::size_t>(prev_index)] * previous_layer.neurons[static_cast<std::size_t>(prev_index)].output;
                }
                sum += neuron.weights.back() * bias;
                neuron.output = sigmoid(sum);
            }

            agent.net.output.resize(static_cast<std::size_t>(agent.net.num_outputs));
            for (int output_index = 0; output_index < agent.net.num_outputs; ++output_index)
            {
                agent.net.output[static_cast<std::size_t>(output_index)] = output_layer.neurons[static_cast<std::size_t>(output_index)].output;
            }
        }

        void apply_chromo_to_agent(Agent &agent, const DoubleVector &data)
        {
            const int copy_count = std::min(agent.num_chromo, static_cast<int>(data.size()));
            for (int index = 0; index < copy_count; ++index)
            {
                agent.chromo[static_cast<std::size_t>(index)] = data[static_cast<std::size_t>(index)];
            }
            sync_chromo_to_net(agent);
        }

        void sync_chromo_to_net(Agent &agent)
        {
            int chromo_index = 0;
            for (Layer &layer : agent.net.layers)
            {
                for (Neuron &neuron : layer.neurons)
                {
                    for (double &weight : neuron.weights)
                    {
                        if (chromo_index < agent.num_chromo)
                        {
                            weight = agent.chromo[static_cast<std::size_t>(chromo_index)];
                        }
                        ++chromo_index;
                    }
                }
            }
        }

        void randomize_chromo(DoubleVector &chromo, int count)
        {
            chromo.assign(static_cast<std::size_t>(count), 0.0);
            for (double &value : chromo)
            {
                value = clamped_rand(16.0);
            }
        }

        int select_parent(const Population &pop) const
        {
            const double total_score = std::accumulate(pop.agents.begin(), pop.agents.end(), 0.0, [](double sum, const Agent &agent)
                                                       { return sum + agent.score; });

            if (total_score <= std::numeric_limits<double>::epsilon())
            {
                return static_cast<int>(rand_unit() * pop.num_agents) % pop.num_agents;
            }

            const double roulette = rand_unit();
            for (const Agent &agent : pop.agents)
            {
                if (roulette > agent.pos_low && roulette <= agent.pos_hi)
                {
                    return agent.index;
                }
            }

            return pop.num_agents - 1;
        }

        DoubleVector breed_chromo(const DoubleVector &mom, const DoubleVector &dad, double cross_rate, int cross_option) const
        {
            DoubleVector baby(mom.size(), 0.0);
            if (mom.empty())
            {
                return baby;
            }

            if (cross_option == 0)
            {
                const std::size_t cut_point = static_cast<std::size_t>(rand_unit() * static_cast<double>(mom.size()));
                for (std::size_t index = 0; index < mom.size(); ++index)
                {
                    if (rand_unit() <= cross_rate)
                    {
                        baby[index] = index < cut_point ? mom[index] : dad[index];
                    }
                    else
                    {
                        baby[index] = mom[index];
                    }
                }
                return baby;
            }

            for (std::size_t index = 0; index < mom.size(); ++index)
            {
                if (rand_unit() <= cross_rate)
                {
                    baby[index] = mom[index];
                }
                else
                {
                    baby[index] = dad[index];
                }
            }
            return baby;
        }

        void mutate_chromo(DoubleVector &chromo, double mutate_rate, double mutate_max, int mutate_option) const
        {
            for (double &gene : chromo)
            {
                if (rand_unit() >= mutate_rate)
                {
                    continue;
                }

                if (mutate_option == 0)
                {
                    gene = clamped_rand(16.0);
                }
                else
                {
                    gene += clamped_rand(16.0) * mutate_max;
                }
            }
        }
    };

    World world;

} // namespace

void info()
{
    world.info();
}

void info_pop()
{
    world.info_pop();
}

void info_net(int pop_index, int agent_index, int level)
{
    world.info_net(pop_index, agent_index, level);
}

bool add_pop(int num_pop_add, double crossover, double mutation, double mutatemax)
{
    return world.add_pop(num_pop_add, crossover, mutation, mutatemax);
}

bool add_agent(int pop_index, int num_agents)
{
    return world.add_agent(pop_index, num_agents);
}

bool add_net(int pop_index, int num_input, int num_layers, int num_neurons, int num_output)
{
    return world.add_net(pop_index, num_input, num_layers, num_neurons, num_output);
}

PopulationInputBatch update(const PopulationInputBatch &data)
{
    return world.update(data);
}

// JS-friendly wrapper that accepts nested JS arrays (populations -> agents -> inputs)
emscripten::val update_js(emscripten::val data)
{
    PopulationInputBatch cdata;
    const unsigned num_pops = data["length"].as<unsigned>();
    cdata.resize(static_cast<std::size_t>(num_pops));

    for (unsigned p = 0; p < num_pops; ++p)
    {
        emscripten::val pop = data[p];
        const unsigned num_agents = pop["length"].as<unsigned>();
        cdata[static_cast<std::size_t>(p)].resize(static_cast<std::size_t>(num_agents));
        for (unsigned a = 0; a < num_agents; ++a)
        {
            emscripten::val agentInput = pop[a];
            const unsigned num_inputs = agentInput["length"].as<unsigned>();
            DoubleVector dv;
            dv.resize(static_cast<std::size_t>(num_inputs));
            for (unsigned i = 0; i < num_inputs; ++i)
            {
                dv[static_cast<std::size_t>(i)] = agentInput[i].as<double>();
            }
            cdata[static_cast<std::size_t>(p)][static_cast<std::size_t>(a)] = std::move(dv);
        }
    }

    PopulationInputBatch result = world.update(cdata);

    emscripten::val out = emscripten::val::array();
    for (std::size_t p = 0; p < result.size(); ++p)
    {
        emscripten::val popArr = emscripten::val::array();
        for (std::size_t a = 0; a < result[p].size(); ++a)
        {
            const DoubleVector &dv = result[p][a];
            emscripten::val agentArr = emscripten::val::array();
            for (std::size_t i = 0; i < dv.size(); ++i)
            {
                agentArr.call<void>("push", dv[i]);
            }
            popArr.call<void>("push", agentArr);
        }
        out.call<void>("push", popArr);
    }
    return out;
}

bool insert_chromo(int pop_index, int agent_index, const DoubleVector &data)
{
    return world.insert_chromo(pop_index, agent_index, data);
}

DoubleVector get_chromo(int pop_index, int agent_index)
{
    return world.get_chromo(pop_index, agent_index);
}

int next_gen(int pop_index, const DoubleVector &data, int cross_option, int mutate_option, int elite_num, int worst_num)
{
    return world.next_gen(pop_index, data, cross_option, mutate_option, elite_num, worst_num);
}

// JS-friendly wrapper for next_gen that accepts optional parameters and a JS array for scores
int next_gen_js(int pop_index, emscripten::val dataVal, emscripten::val cross_option_val, emscripten::val mutate_option_val, emscripten::val elite_num_val, emscripten::val worst_num_val)
{
    DoubleVector data;
    if (!dataVal.isUndefined() && dataVal.typeOf().as<std::string>() == "object")
    {
        const unsigned len = dataVal["length"].as<unsigned>();
        data.resize(static_cast<std::size_t>(len));
        for (unsigned i = 0; i < len; ++i)
        {
            data[static_cast<std::size_t>(i)] = dataVal[i].as<double>();
        }
    }

    const int cross_option = cross_option_val.isUndefined() ? 1 : cross_option_val.as<int>();
    const int mutate_option = mutate_option_val.isUndefined() ? 1 : mutate_option_val.as<int>();
    const int elite_num = elite_num_val.isUndefined() ? 0 : elite_num_val.as<int>();
    const int worst_num = worst_num_val.isUndefined() ? 0 : worst_num_val.as<int>();

    return world.next_gen(pop_index, data, cross_option, mutate_option, elite_num, worst_num);
}

EMSCRIPTEN_BINDINGS(nnetga_module)
{
    emscripten::register_vector<double>("VectorDouble");
    emscripten::register_vector<DoubleVector>("VectorVectorDouble");
    emscripten::register_vector<AgentInputBatch>("VectorVectorVectorDouble");

    emscripten::function("info", &info);
    emscripten::function("info_pop", &info_pop);
    emscripten::function("info_net", &info_net);
    emscripten::function("add_pop", &add_pop);
    emscripten::function("add_agent", &add_agent);
    emscripten::function("add_net", &add_net);
    emscripten::function("update", &update);
    emscripten::function("update_js", &update_js);
    emscripten::function("insert_chromo", &insert_chromo);
    emscripten::function("get_chromo", &get_chromo);
    emscripten::function("next_gen", &next_gen);
    emscripten::function("next_gen_js", &next_gen_js);
}