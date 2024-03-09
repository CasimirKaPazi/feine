-- Initialize the neural grid
function love.load()
	seed = os.time()
	math.randomseed(seed)
	local chunk = love.filesystem.load("functions.lua")
	chunk()

	-- general
	GRID_WIDTH = 100
	GRID_HEIGHT = 100
	grid = {}
	CELLSIZE = 3 -- pixel on screen
	-- affecting cells
	RANGE = 2 -- how far from origin weights are drawn
	N_WEIGHTS = (RANGE * 2 + 1)^2 - 1 -- number of connections increases exponentially with range
	MIN_LEARNING = 0.0001 -- found through trial and error. Affects the ratio between learning and evolution.
	N_MUTATIONS = 5 -- mutations per 100x100 area per time step
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

	love.window.setTitle("Feine")
	love.window.setMode(GRID_WIDTH * CELLSIZE, GRID_HEIGHT * CELLSIZE)
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
						local nI = i + x
						local nJ = j + y
						nI, nJ = wrapAroundGrid(nI, nJ)
						sum = sum + grid[nI][nJ].act * grid[i][j].weights[w]
						w = w+1
					end
				end
			end
			-- Save old activation for later use in learning
			grid[i][j].past2_act = grid[i][j].past_act
			grid[i][j].past_act = currentact
			-- Update activation using ReLU
			sum = math.min(math.max(0, sum), 10000)
--			sum = math.min(10000, math.max(0, (sum*3 + currentact*grid[i][j].memory)/(1*3+grid[i][j].memory)))
			average_sum = average_sum + sum + grid[i][j].past_act + grid[i][j].past2_act
			grid[i][j].new_act = sum
		end
	end
	average_sum = average_sum / GRID_WIDTH / GRID_HEIGHT
end

-- Learning
function updateWeights()
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			-- Update the grid to new state
			grid[i][j].act = grid[i][j].new_act
			-- Update weights to minimize surprise, using the qudratic loss funtion
			local err = -2 * (grid[i][j].act - grid[i][j].past_act)
			local learningrate = (grid[i][j].learning^4 + MIN_LEARNING)
			grid[i][j].err = err
			for w = 1, N_WEIGHTS do
				grid[i][j].weights[w] = grid[i][j].weights[w] + learningrate * err * math.tanh(grid[i][j].act)
			end
		end
	end
end

-- Reproduction
function updateReproduction()
	average_fitness = 0
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			local nI = i
			local nJ = j
			local r = math.random(4)
			if r == 1 then
				nI = (i + 1)
			elseif r == 2 then
				nI = (i - 1)
			elseif r == 3 then
				nJ = (j + 1)
			elseif r == 4 then
				nJ = (j - 1)
			end
			nI, nJ = wrapAroundGrid(nI, nJ)
			local fitness = 0
				-- compare how well the cell and neighbor predict each other
				-- when the difference is 0 this becomes 1
				- 2^(-1*math.abs(grid[i][j].act - grid[nI][nJ].past_act))
				+ 2^(-1*math.abs(grid[nI][nJ].act - grid[i][j].past_act))
				- math.tanh(grid[nI][nJ].act + grid[nI][nJ].past_act) * (1-math.tanh(math.abs(grid[nI][nJ].err)))
				+ math.tanh(grid[i][j].act + grid[i][j].past_act) * (math.tanh(math.abs(grid[i][j].err)))
			if grid[nI][nJ].act == 0 then fitness = 0 end
			if fitness < 0 or (grid[i][j].past_act * grid[i][j].past_act * grid[i][j].past2_act)^(1/3) >= 9000 then
				copyGenes(i, j, nI, nJ)
			else
				copyGenes(i, j, i, j)
			end
			grid[i][j].fitness = fitness
			average_fitness = average_fitness + fitness
		end
	end
	average_fitness = average_fitness / GRID_WIDTH / GRID_HEIGHT
	-- use new weights and genes 
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			for w = 1, N_WEIGHTS do
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
		-- Introduce some noise to restart dead simulations and to select for more robust patterns
		grid[i][j].act = grid[i][j].act + math.random()/50
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
		m = math.random(1,4)
		if m == 1 then -- invert random weight
			local w = math.random(1, N_WEIGHTS)
			grid[i][j].weights[w] = mutate(grid[i][j].weights[w]+0.5)-0.5
		elseif m == 2 then -- copy one weight onto another
			local w = math.random(1, N_WEIGHTS)
			local k = math.random(1, N_WEIGHTS)
			grid[i][j].weights[w] = grid[i][j].weights[k]
		elseif m == 3 then -- invert random weight
			local w = math.random(1, N_WEIGHTS)
			grid[i][j].weights[w] = -1* grid[i][j].weights[w]
		elseif m >= 4 then -- merge genes with direct neighbor
			local x = math.random(-1,1)
			local y = math.random(-1,1)
			if math.abs(x) + math.abs(y) == 1 then
				local nI, nJ = wrapAroundGrid(i + x, j + y)
				if ( math.abs(grid[i][j].color1 - grid[nI][nJ].color1)
					+ math.abs(grid[i][j].memory - grid[nI][nJ].memory)
					+ math.abs(grid[i][j].learning - grid[nI][nJ].learning) )/3 
					< (grid[i][j].color1 + grid[nI][nJ].color1)/2 then
					mixGenes(i, j, nI, nJ)
				end
			end
		end
		grid[i][j].learning = math.max(0, grid[i][j].learning)
		grid[i][j].memory = math.max(0, math.min(1, grid[i][j].memory))
		grid[i][j].color1 = math.max(0, math.min(1, grid[i][j].color1))
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
--						copyGenes(i, j, cellX, cellY)
--						grid[i][j].act = math.random()
--						grid[i][j].past_act = math.random()
--						grid[i][j].past2_act = math.random()
						for w = 1, N_WEIGHTS do
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
		shiftMap(1, 0)
	elseif love.keyboard.isDown( "left" ) then
		shiftMap(-1, 0)
	end
	if love.keyboard.isDown( "down" ) then
		shiftMap(0, 1)
	elseif love.keyboard.isDown( "up" ) then
		shiftMap(0, -1)
	end

--	if generation == 5000 then pause = true end -- For testing

end

-- Show on screen
function love.draw()
	if view_mode == 0 then -- Grid hidden
		love.graphics.setColor(1, 1, 1)
		love.graphics.printf("Generation: "..generation, 0, 0, GRID_WIDTH)
		love.graphics.printf("Average: "..math.floor(average_sum), 0, 24, GRID_WIDTH)
		love.graphics.printf("Fitness: "..math.floor(average_fitness*10000)/10000, 0, 48, GRID_WIDTH)
		return
	end
    for i = 1, GRID_WIDTH do
        for j = 1, GRID_HEIGHT do
            local x = (i - 1) * CELLSIZE
            local y = (j - 1) * CELLSIZE
			local red = 0
			local green = 0
			local blue = 0
			if view_mode == 1 then -- activation with slight indication of genes
		        red = math.tanh(grid[i][j].act - grid[i][j].err/8)
		        green = math.tanh(grid[i][j].past_act - grid[i][j].err/8)
		        blue = math.tanh(grid[i][j].act + grid[i][j].past_act + grid[i][j].past2_act/2)
			elseif view_mode == 2 then -- error
		        red = math.tanh(math.abs(grid[i][j].err))
		        green = math.tanh(math.abs(grid[i][j].err))
		        blue = math.tanh(math.abs(grid[i][j].err))
			elseif view_mode == 3 then -- focus learning
		        red = math.tanh(grid[i][j].past_act)/16
		        green = math.tanh(grid[i][j].act)/16
		        blue = math.tanh(grid[i][j].learning*2)
			elseif view_mode == 4 then -- focus color1
		        red = math.tanh(grid[i][j].color1*2)
		        green = math.tanh(grid[i][j].past_act)/16
		        blue = math.tanh(grid[i][j].act)/16
			elseif view_mode == 5 then -- focus memory
		        red = math.tanh(grid[i][j].act)/16
		        green = math.tanh(grid[i][j].memory*2)
		        blue = math.tanh(grid[i][j].past_act)/16
			elseif view_mode == 6 then -- focus memory
		        red = math.tanh(grid[i][j].color1*2)
		        green = math.tanh(grid[i][j].memory*2)
		        blue = math.tanh(grid[i][j].learning*2)
			elseif view_mode == 7 then -- focus memory
		        red = math.tanh(grid[i][j].color1/3 + grid[i][j].act/2 - grid[i][j].err/16)
		        green = math.tanh(grid[i][j].memory/3 + grid[i][j].past_act/2 - grid[i][j].err/16)
		        blue = math.tanh(grid[i][j].learning/3 + grid[i][j].act/2 + grid[i][j].past_act/4 + grid[i][j].past2_act/8)
			end
            love.graphics.setColor(red, green, blue)
            love.graphics.rectangle("fill", x, y, CELLSIZE, CELLSIZE)
        end
    end
end

function love.quit()
	print("The simulation ran for "..generation.." generations. Seed was: "..seed..".")
end
