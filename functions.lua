-- Function to normalize indices and ensure looping around the grid
function wrapAroundGrid(i, j)
    return ((i - 1) % GRID_WIDTH) + 1, ((j - 1) % GRID_HEIGHT) + 1
end

-- Move the grid
function shiftMap(x, y)
	if x >= 0 and y >= 0 then
		Ai, Zi, Di = 1, GRID_WIDTH, 1
		Aj, Zj, Dj = 1, GRID_HEIGHT, 1
	else
		Ai, Zi, Di = GRID_WIDTH, 1, -1
		Aj, Zj, Dj = GRID_HEIGHT, 1, -1
	end
	for i = Ai, Zi, Di do
		for j = Aj, Zj, Dj do
			nI, nJ = wrapAroundGrid(i+x, j+y)
			copyGenes(i, j, nI, nJ)
			grid[i][j].act = grid[nI][nJ].act
			grid[i][j].past_act = grid[nI][nJ].past_act
			grid[i][j].past2_act = grid[nI][nJ].past2_act
			grid[i][j].err = grid[nI][nJ].err
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
		color1 = math.random(),
		memory = math.random(),
		learning = math.random(),
		weights = {}  -- Initialize weights for neighbors within range
	}
	for x = -RANGE, RANGE do
		for y = -RANGE, RANGE do
			if x ~= 0 or y ~= 0 then
				local random = (math.random() - 0.5) * 2 -- between -1 and 1
				random = random^3 * (12/RANGE)
				table.insert(
					-- Divide by |x|+|y| to give more weight to closer neighbors
					grid[i][j].weights,
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
		grid[i][j].weights[w] = grid[nI][nJ].weights[w]
 	end
	grid[i][j].color1 = grid[nI][nJ].color1
	grid[i][j].memory = grid[nI][nJ].memory
	grid[i][j].learning = grid[nI][nJ].learning
end

function mixGenes(i, j, nI, nJ)
	for w = 1, N_WEIGHTS do
		if math.random(2) == 1 then
			grid[i][j].weights[w] = grid[nI][nJ].weights[w]
		end
	end
	if math.random(2) == 1 then
		grid[i][j].color1 = grid[nI][nJ].color1
	end
	if math.random(2) == 1 then
		grid[i][j].memory = grid[nI][nJ].memory
	end
	if math.random(2) == 1 then
		grid[i][j].learningrate = grid[nI][nJ].learningrate
	end
end

-- Calculate mutation for a gene
function mutate(x)
	-- This assumes that the variation of x is centerd around 0.5
	return x + (math.random()-0.5) * 0.5 * (x+0.5)
end
