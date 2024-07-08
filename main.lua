-- Initialize the neural grid
function love.load()
	seed = os.time()
	math.randomseed(seed)
	local chunk = love.filesystem.load("functions.lua")
	chunk()

	-- general
	GRID_WIDTH = 100
	GRID_HEIGHT = 100
	CELLSIZE = 3 -- pixel on screen
	grid = {}
	weight_pos = {}
	-- affecting cells
	RANGE = 2 -- how far from origin weights are drawn
	N_WEIGHTS = (RANGE * 2 + 1)^2 - 1 -- number of connections increases exponentially with range
	MIN_LEARNING = 0.0001 -- found through trial and error. Affects the ratio between learning and evolution.
	N_MUTATIONS = 5 -- mutations per 100x100 area per time step
	MAX_ACT = 10000
	average_sum = 0
	average_fitness = 0
	-- for tools
	generation = 0 -- count generations
	pause = false -- pause simulation
	step = false -- do one time step even when paused
	mousex, mousey = 0, 0 -- track past mouse position
	hide_mouse = false
	view_mode = 1
	prev_view_mode = 0
	shiftX = 0
	shiftY = 0

	love.window.setTitle("Feine")
	love.window.setMode(GRID_WIDTH * CELLSIZE, GRID_HEIGHT * CELLSIZE)

	-- Generate a table with weight number and coordinates
	gen_weight_pos()
	-- Seed the grid with random weights and genes
	initGrid()
end

-- Thinking
function updateActivation()
	-- Update neuron activations and weights
	average_sum = 0
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			local currentact = grid[i][j].act
			-- Thinking and learning
			-- Calculate the weighted sum of inputs
			local sum = 0
			local w = 1
			for x = -RANGE, RANGE do
				for y = -RANGE, RANGE do
					if x ~= 0 or y ~= 0 then
						local nI, nJ = wrapAroundGrid(i + x, j + y)
						sum = sum + grid[nI][nJ].past_act * grid[i][j].past_weights[w] * grid[i][j].memory
						sum = sum + grid[nI][nJ].act * grid[i][j].weights[w]
						w = w+1
					end
				end
			end
			-- Save old activation for later use in learning
			grid[i][j].past2_act = grid[i][j].past_act
			grid[i][j].past_act = currentact
			-- Update activation using ReLU
			-- Reduce activation a litte to pose a challenge for the organisms
			sum = math.max(0, sum)
			if sum >= MAX_ACT then sum = 0 end
			average_sum = average_sum + sum
			grid[i][j].new_act = sum
		end
	end
	average_sum = average_sum / GRID_WIDTH / GRID_HEIGHT
end

-- Learning
function updateWeights()
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			local c = grid[i][j]
			-- Update the grid to new state
			c.act = c.new_act
			-- Update weights to minimize surprise, using the qudratic loss funtion
			local err = -2 * (c.act - c.past_act)
			local learningrate = (c.learning^4 + MIN_LEARNING)
			c.past_err = c.err
			c.err = err
			for w = 1, N_WEIGHTS do
				local nI, nJ = wrapAroundGrid(i + weight_pos[w].x, j + weight_pos[w].y)
				c.past_weights[w] = c.past_weights[w] +
					learningrate * err * c.act * grid[nI][nJ].past_act * c.memory
				c.past_weights[w] = loop(c.past_weights[w], -1, 1)
				c.weights[w] = c.weights[w] +
					learningrate * err * c.act * grid[nI][nJ].act
				c.weights[w] = loop(c.weights[w], -1, 1)
			end
		end
	end
end
 

-- When activation is close to average of neighbors, change activation
function updateDiffusion()
	average_fitness = 0
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			local neighbors = {
				{x = i + 1, y = j, p = 0.147},
				{x = i - 1, y = j, p = 0.147},
				{x = i, y = j + 1, p = 0.147},
				{x = i, y = j - 1, p = 0.147},
				{x = i + 1, y = j + 1, p = 0.103},
				{x = i - 1, y = j + 1, p = 0.103},
				{x = i + 1, y = j - 1, p = 0.103},
				{x = i - 1, y = j - 1, p = 0.103}
			}
			local diff = 0
			for _, neighbor in ipairs(neighbors) do
				local nI, nJ = wrapAroundGrid(neighbor.x, neighbor.y)
				local n = grid[nI][nJ]
				diff = diff + math.tanh(n.act) * neighbor.p
			end
			local activation = grid[i][j].act--(grid[i][j].act * 3 + (diff - grid[i][j].act) * grid[i][j].color1)/(3 + grid[i][j].color1)
			local change = math.tanh(grid[i][j].act) - diff
			if change == 0 then
				change = (math.random()-0.5)/1000
			else
				change = math.tanh(1/change)
			end
			activation = math.max(0, activation + 0.01 * math.tanh(grid[i][j].act) * change * (grid[i][j].color1 + MIN_LEARNING))
			grid[i][j].new_act = activation
		end
	end
end

-- Reproduction
function updateReproduction()
	average_fitness = 0
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			grid[i][j].act = grid[i][j].new_act
			local neighbors = {
				{x = i + 1, y = j, p = 1},
				{x = i - 1, y = j, p = 1},
				{x = i, y = j + 1, p = 1},
				{x = i, y = j - 1, p = 1},
				{x = i + 1, y = j + 1, p = 0.71},
				{x = i - 1, y = j + 1, p = 0.71},
				{x = i + 1, y = j - 1, p = 0.71},
				{x = i - 1, y = j - 1, p = 0.71}
			}

			-- Neighbors compete with center
			local bestNeighbors = {{x = i, y = j}}
			-- Penalize static patterns and high activation
			local c = grid[i][j]
			local selfFitness = get_fitness(c)
			local bestFitness = selfFitness

			-- Iterate through neighbors to find the best fit
			for _, neighbor in ipairs(neighbors) do
				local nI, nJ = wrapAroundGrid(neighbor.x, neighbor.y)
				local n = grid[nI][nJ]
				local fitness = get_fitness(n) * neighbor.p
						* (	math.tanh( math.abs(c.act - c.past_act + 0.5) * (c.act + n.past_act + 0.5) )/
						math.tanh( math.abs(c.act - n.past_act + 0.5) * (c.act + c.past_act + 0.5) ) )^2
				if fitness > bestFitness then
					bestFitness = fitness
					bestNeighbors = {{x = nI, y = nJ}}
				elseif fitness == bestFitness then
					table.insert(bestNeighbors, {x = nI, y = nJ})
				end
			end

			-- Randomly select one of the best-fit neighbors
			if bestFitness > math.max(0, selfFitness * (1+c.cooldown) + c.cooldown) then
				local parent_pos = bestNeighbors[math.random(1, #bestNeighbors)]
				copyGenes(i, j, parent_pos.x, parent_pos.y)
				c.cooldown = c.cooldown + 1
			else
				copyGenes(i, j, i, j)
				c.cooldown = c.cooldown * 0.75
			end

			c.fitness = bestFitness
			average_fitness = average_fitness + bestFitness
		end
	end
	average_fitness = average_fitness / GRID_WIDTH / GRID_HEIGHT
	-- use new weights and genes 
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			for w = 1, N_WEIGHTS do
				grid[i][j].past_weights[w] = grid[i][j].new_past_weights[w]
				grid[i][j].weights[w] = grid[i][j].new_weights[w]
		 	end
			grid[i][j].color1 = grid[i][j].new_color1
			grid[i][j].memory = grid[i][j].new_memory
			grid[i][j].learning = grid[i][j].new_learning
		end
	end
end

-- Mutation
function updateMutation()
	-- Pick several cells at random to mutate
	for n = 1, math.ceil(N_MUTATIONS * GRID_WIDTH/100 * GRID_HEIGHT/100) do
		local i, j = math.random(1, GRID_WIDTH), math.random(1, GRID_HEIGHT)
		-- Introduce some noise to restart dead simulations
		if average_sum * math.random() < 1 then
			grid[i][j].act = grid[i][j].act + math.random()/50
		end
		mutuate(i, j)
	end
end

-- Key events
function love.keypressed(key, scancode, isrepeat)
	if key == "space" then -- toggle pause simulation
		pause = not(pause)
	elseif key == "a" then -- continue paused simulation one time step
		step = true
	elseif key == "0" and view_mode ~= 0 then
		prev_view_mode = view_mode
		view_mode = 0
	elseif key == "1" and view_mode ~= 1 then
		prev_view_mode = view_mode
		view_mode = 1
	elseif key == "2" and view_mode ~= 2 then
		prev_view_mode = view_mode
		view_mode = 2
	elseif key == "3" and view_mode ~= 3 then
		prev_view_mode = view_mode
		view_mode = 3
	elseif key == "4" and view_mode ~= 4 then
		prev_view_mode = view_mode
		view_mode = 4
	elseif key == "5" and view_mode ~= 5 then
		prev_view_mode = view_mode
		view_mode = 5
	elseif key == "6" and view_mode ~= 6 then
		prev_view_mode = view_mode
		view_mode = 6
	elseif key == "7" and view_mode ~= 7 then
		prev_view_mode = view_mode
		view_mode = 7
	elseif key == "tab" then -- Switch to last used viewing mode
		local swap = prev_view_mode
		prev_view_mode = view_mode
		view_mode = swap
	end
end

-- Mouse events
function love.mousepressed(x, y, button, istouch, presses)
	-- Print learning rate of pointed cell
	if button == 1 then -- Left mouse button clicked
		local cellX = math.floor(x / CELLSIZE) + 1
		local cellY = math.floor(y / CELLSIZE) + 1
		if cellX >= 1 and cellX <= GRID_WIDTH and cellY >= 1 and cellY <= GRID_HEIGHT then
			if love.keyboard.isDown( "lshift" ) then
				-- Replace all cells with genes of pointed
				for i = 1, GRID_WIDTH do
					for j = 1, GRID_HEIGHT do
						for w = 1, N_WEIGHTS do
							grid[i][j].past_weights[w] = grid[cellX][cellY].past_weights[w]
							grid[i][j].weights[w] = grid[cellX][cellY].weights[w]
					 	end
						grid[i][j].color1 = grid[cellX][cellY].color1
						grid[i][j].memory = grid[cellX][cellY].memory
						grid[i][j].learning = grid[cellX][cellY].learning
					end
				end
			else
				-- Print information about the cell to terminal
				print("act = ",grid[cellX][cellY].act)
				print("err = ",grid[cellX][cellY].err)
				print("fit = ",grid[cellX][cellY].fitness)
				print("color1 = ",grid[cellX][cellY].color1)
				print("memory = ",grid[cellX][cellY].memory)
				print("learning = ",grid[cellX][cellY].learning)
				for w = 1, N_WEIGHTS do
					print("past weight "..w.." = ",grid[cellX][cellY].past_weights[w])
					print("weight "..w.." = ",grid[cellX][cellY].weights[w])
				end
			end
		end
	end
end

-- Callbacks for each time step
function love.update(dt)
	-- Main functions
	if not pause or step then
		updateActivation() -- think
		updateWeights() -- learn
		updateReproduction() -- reproduce
		updateMutation() -- diversify
		generation = generation + 1
		step = false
	end

	-- Hide the mouse when not moving
	if hide_mouse then
		local mx, my = love.mouse.getPosition( )
		if mx == mousex and my == mousey then
			love.mouse.setVisible( false )
		else
			love.mouse.setVisible( true )
		end
		mousex, mousey = mx, my
	end

	-- randomize cells by right click
	if love.mouse.isDown( 2 ) then
		local x, y = love.mouse.getPosition( )
		local cellX = math.floor(x / CELLSIZE) + 1
		local cellY = math.floor(y / CELLSIZE) + 1
		if cellX >= 1 and cellX <= GRID_WIDTH and cellY >= 1 and cellY <= GRID_HEIGHT then
			initCell(cellX, cellY)
		end
	end

	-- Move the grid with arrow keys
	if love.keyboard.isDown( "right" ) then
		shiftX = ((shiftX - 1 - 1) % GRID_WIDTH) + 1
	elseif love.keyboard.isDown( "left" ) then
		shiftX = ((shiftX + 1 - 1) % GRID_WIDTH) + 1
	end
	if love.keyboard.isDown( "down" ) then
		shiftY = ((shiftY - 1 - 1) % GRID_WIDTH) + 1
	elseif love.keyboard.isDown( "up" ) then
		shiftY = ((shiftY + 1 - 1) % GRID_WIDTH) + 1
	end

--	if generation == 5000 then pause = true end -- For testing
--	if generation == 1000 then print("1000 generations took "..os.time() - seed.." seconds") end
end

-- Show on screen
function love.draw()
	if view_mode == 0 then -- Grid hidden
		love.graphics.setColor(1, 1, 1)
		love.graphics.printf("Generation: "..generation, 0, 0, GRID_WIDTH)
		love.graphics.printf("Activation: "..math.floor(average_sum*1000)/1000, 0, 24, GRID_WIDTH)
		love.graphics.printf("Fitness: "..math.floor(average_fitness*1000)/1000, 0, 48, GRID_WIDTH)
		return
	end
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			local x = i + shiftX
			local y = j + shiftY
			x, y = wrapAroundGrid(x, y)
			x = (x - 1) * CELLSIZE
			y = (y - 1) * CELLSIZE
			local red = 0
			local green = 0
			local blue = 0
			if view_mode == 1 then -- activation with slight indication of genes
				red = grid[i][j].act - grid[i][j].err/8
				green = grid[i][j].past_act - grid[i][j].err/8
				blue = grid[i][j].act + grid[i][j].past_act + grid[i][j].past2_act/2
			elseif view_mode == 2 then -- error
				red = math.abs(grid[i][j].err)
				green = grid[i][j].fitness^(1/2)
				blue = grid[i][j].cooldown
			elseif view_mode == 3 then -- focus learning
				red = math.tanh(grid[i][j].past_act)/8
				green = math.tanh(grid[i][j].act)/8
				blue = (grid[i][j].learning^(1/2))
			elseif view_mode == 4 then -- focus color1
				red = grid[i][j].color1
				green = math.tanh(grid[i][j].past_act)/8
				blue = math.tanh(grid[i][j].act)/8
			elseif view_mode == 5 then -- focus memory
				red = math.tanh(grid[i][j].act)/8
				green = grid[i][j].memory
				blue = math.tanh(grid[i][j].past_act)/8
			elseif view_mode == 6 then -- genes
				red = grid[i][j].color1
				green = grid[i][j].memory
				blue = grid[i][j].learning^(1/2)
			elseif view_mode == 7 then -- genes + activation
				red = grid[i][j].color1/3 + (grid[i][j].act/2 - grid[i][j].err/16)
				green = grid[i][j].memory/3 + (grid[i][j].past_act/2 - grid[i][j].err/16)
				blue = grid[i][j].learning^(1/2)/3 + (grid[i][j].act/2 + grid[i][j].past_act/4 + grid[i][j].past2_act/8)
			end
			love.graphics.setColor(red, green, blue)
			love.graphics.rectangle("fill", x, y, CELLSIZE, CELLSIZE)
		end
	end
end

function love.quit()
	print("The simulation ran for "..generation.." generations. Seed was: "..seed..".")
end
