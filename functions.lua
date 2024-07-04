-- loop x to be between min and max
function loop(x, min, max)
    if x < min then
        x = math.abs(x - min) + min
    end
    if x > max then
        x = max - math.abs(x - max)
    end
    return x
end

-- Function to normalize indices and ensure looping around the grid
function wrapAroundGrid(i, j)
    return ((i - 1) % GRID_WIDTH) + 1, ((j - 1) % GRID_HEIGHT) + 1
end

function gen_weight_pos()
	for x = -RANGE, RANGE do
		for y = -RANGE, RANGE do
			if x ~= 0 or y ~= 0 then
				table.insert(weight_pos, {x = x, y = y})
--				print("weight_pos nr: "..#weight_pos.." x: "..x.." y: "..y)
			end
		end
	end
end

function initCell(i, j)
	local init = math.random()
	grid[i][j] = {
		act = init,  -- Neuron activation
		past_act = init, -- Activation of last step
		past2_act = init, -- Activation of two steps past
		new_act = 0,
		err = 0,
		past_err = 0,
		fitness = 0,
		cooldown = 1,
		color1 = math.random(),
		memory = math.random(),
		learning = math.random()^4,
		weights = {},  -- Initialize weights for neighbors within range
		past_weights = {},  -- Initialize weights for neighbors within range
		new_color1 = math.random(),
		new_memory = math.random(),
		new_learning = math.random()^4,
		new_weights = {},
		new_past_weights = {}
	}
	for x = -RANGE, RANGE do
		for y = -RANGE, RANGE do
			if x ~= 0 or y ~= 0 then
				-- Weights for past states
				table.insert(grid[i][j].past_weights, 0)
				table.insert(grid[i][j].new_past_weights, 0)
				-- Weights for current states
				local random = (math.random() - 0.5) * 2 -- between -1 and 1
				random = random^3 * (12/RANGE)
				table.insert(
					-- Divide by |x|+|y| to give more weight to closer neighbors
					grid[i][j].weights,
					random / (math.abs(x) + math.abs(y))
				)
				table.insert(
					-- Divide by |x|+|y| to give more weight to closer neighbors
					grid[i][j].new_weights,
					random / (math.abs(x) + math.abs(y))
				)
			end
		end
	end
end

-- Create a cell with random values
function initGrid()
	for i = 1, GRID_WIDTH do
		grid[i] = {}
		for j = 1, GRID_HEIGHT do
			initCell(i, j)
		end
	end
end

-- Evaluate fitness
function get_fitness(self)
	local fitness =
	-- Select for alive patterns
	math.tanh(self.act)
	-- Don't escalate activation
	* math.tanh(MAX_ACT - self.act^2)
	-- Penalize static patterns and high activation
	- math.tanh(math.max(self.act, self.past_act, self.past2_act))^2
	- math.tanh(math.min(self.act, self.past_act, self.past2_act)*2)^2
	-- Prevent blurr
	- math.tanh(math.abs(self.err) * math.abs(self.past_err))/8
	return fitness
end

function mutuate(i, j)
	-- Modify genes
	local m = math.random(1,4)
	if m == 1 then -- color1
		grid[i][j].color1 = mutate(grid[i][j].color1)
	elseif m == 2 then -- memory
		grid[i][j].memory = mutate(grid[i][j].memory)
	elseif m == 3 then -- leanring rate
		grid[i][j].learning = mutate(grid[i][j].learning)
	elseif m == 4 then -- leanring rate
		grid[i][j].color1 = mutate(grid[i][j].color1)
		grid[i][j].memory = mutate(grid[i][j].memory)
		grid[i][j].learning = mutate(grid[i][j].learning)
	end
	-- Modify weights
	m = math.random(1,2)
	if m == 1 then -- mutate random weight
		local w = math.random(1, N_WEIGHTS)
		grid[i][j].past_weights[w] = mutate(grid[i][j].past_weights[w]+0.5)-0.5
		grid[i][j].weights[w] = mutate(grid[i][j].weights[w]+0.5)-0.5
	elseif m == 2 then -- copy one weight onto another
		local w = math.random(1, N_WEIGHTS)
		local k = math.random(1, N_WEIGHTS)
		grid[i][j].past_weights[w] = grid[i][j].past_weights[k]
		grid[i][j].weights[w] = grid[i][j].weights[k]
	end
	grid[i][j].learning = loop(grid[i][j].learning, 0, 1)
	grid[i][j].memory = loop(grid[i][j].memory, 0, 1)
	grid[i][j].color1 = loop(grid[i][j].color1, 0, 1)
end

-- Copy Genes from another cell
function copyGenes(i, j, nI, nJ)
	for w = 1, N_WEIGHTS do
		grid[i][j].new_past_weights[w] = grid[nI][nJ].past_weights[w]
		grid[i][j].new_weights[w] = grid[nI][nJ].weights[w]
 	end
	grid[i][j].new_color1 = grid[nI][nJ].color1
	grid[i][j].new_memory = grid[nI][nJ].memory
	grid[i][j].new_learning = grid[nI][nJ].learning
end

function mixGenes(i, j, nI, nJ)
	for w = 1, N_WEIGHTS do
		if math.random(2) == 1 then
			grid[i][j].new_past_weights[w] = grid[nI][nJ].past_weights[w]
		end
		if math.random(2) == 1 then
			grid[i][j].new_weights[w] = grid[nI][nJ].weights[w]
		end
	end
	if math.random(2) == 1 then
		grid[i][j].new_color1 = grid[nI][nJ].color1
	end
	if math.random(2) == 1 then
		grid[i][j].new_memory = grid[nI][nJ].memory
	end
	if math.random(2) == 1 then
		grid[i][j].new_learning = grid[nI][nJ].learning
	end
end

-- Calculate mutation for a gene
function mutate(x)
	-- This assumes that the variation of x is centerd around 0.5
	return x + (math.random()-0.5) * 0.5 * (x+0.5)
end
