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
		end
	end
end

function initCell(i, j)
	grid[i][j] = {
		act = 0,  -- Neuron activation
		past_act = 0, -- Activation of last step
		past2_act = 0, -- Activation of two steps past
		new_act = 0,
		err = 0,
		color1 = 0,
		memory = 0,
		learning = math.random(),--MIN_LEARNING, -- bias learning to smaller values
		weights = {}  -- Initialize weights for neighbors within range
	}
	for x = -RANGE, RANGE do
		for y = -RANGE, RANGE do
			if x ~= 0 or y ~= 0 then
				-- Divide by |x|+|y| to give more weight to closer neighbors
				local random = (math.random() - 0.5) * 2 -- between -1 and 1
				random = random^3 * (12/RANGE)
				table.insert(
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

-- Calculate mutation for a gene
function mutate(x)
	-- This assumes that the variation of x is centerd around 0.5
	return x + (math.random()-0.5) * 0.5 * (x+0.5)
end
