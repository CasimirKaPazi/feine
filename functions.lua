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
		fitness = 0,
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
		grid[i][j].new_learningrate = grid[nI][nJ].learningrate
	end
end

-- Calculate mutation for a gene
function mutate(x)
	-- This assumes that the variation of x is centerd around 0.5
	return x + (math.random()-0.5) * 0.5 * (x+0.5)
end
