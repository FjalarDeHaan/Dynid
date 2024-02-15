module Dynid

using Agents, Graphs
using Random, Distributions
using LinearAlgebra
using GLMakie, InteractiveDynamics

@agent Spin GridAgent{2} begin
    spin::Int # Taking values in {-1, +1}.
end

@agent Vectofer GraphAgent begin
    identities::Vector{Float64}
end

function spin_colour(spin::Spin)
    if spin.spin == +1
        return "#111111"
    else
        return "#cccccc"
    end
end

function hamiltonian(agent, model)
    neighbours = nearby_ids(agent, model)
    if !isempty(neighbours)
        return -.5 * sum( [ agent.identities ⋅ model[neighbour].identities
                            for neighbour ∈ neighbours ] )
    else
        return 0.0
    end
end

hamiltonian(model) = sum( [ hamiltonian(agent, model)
                            for agent in values(model.agents) ] )

function agent_colour(agent::Vectofer, model)
    h = hamiltonian(agent, model)
    k = nearby_agents(agent, model) |> length
    return (:black, 1 + model.ndims*(h/k))
end

function initialise(
                   ; ndims=10
                   , nvertices=10
                   , seed=nothing
                   )
    # If no seed provided, get the pseudo-randomness from device.
    if isnothing(seed)
        rng = Xoshiro(rand(RandomDevice(), 0:2^16))
    else
        rng = Xoshiro(seed)
    end
    # Prepare the network.
    network = barabasi_albert(nvertices, 3, 2)
    space = GraphSpace(network)
    # Set up the ABM.
    model = AgentBasedModel( Vectofer, space
                           ; rng = rng
                           , properties = Dict(:ndims=>ndims) )
    # Put the agents on the network.
    for i ∈ 1:nvertices
        identity = rand(rng, Dirichlet(ndims, 1.0))
        agent = Vectofer(i, i, identity)
        add_agent_pos!(agent, model)
    end
    # Deliver.
    return model
end

function agent_step!(agent, model)
    # Find neighbours of agent.
    neighbours = nearby_ids(agent, model)
    # Find average identities of the agent's neighbours.
    if !isempty(neighbours)
        neighbourids = [model[neighbour].identities for neighbour in neighbours]
        meanidentity = sum(neighbourids) / length(neighbourids)
        agent.identities = meanidentity
    end
end


function ising(
              ; dimensions = (100, 100)
              , seed = 999
              , β = 1 )
    lattice = GridSpaceSingle(dimensions, periodic=false)
    model = AgentBasedModel( Spin
                           , lattice
                           , scheduler = Schedulers.randomly
                           , rng = MersenneTwister(seed) )
    for i ∈ 1:prod(dimensions)
        add_agent_single!(model, rand(model.rng, [-1, +1]))
    end
    properties = Dict( :β => β )
    return model
end

function dash(model=nothing)
    if isnothing(model)
        model = initialise()
    end
    agent_colour(agent) = agent_colour(agent, model)
    fig, ax, abmp = abmplot(model; ac = agent_colour)
    display(fig)
    return fig, ax, abmp
end

function isingdash(model=nothing)
    if isnothing(model)
        model = ising()
    end
    fig, ax, abmp = abmplot(model; ac = spin_colour, as = 5)
    display(fig)
    return fig, ax, abmp
end


end # Module Dynid.
