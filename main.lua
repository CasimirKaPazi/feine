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
	N_MUTATIONS = 10 -- mutations per 100x100 area per time step
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
			grid[i][j].new_act = math.max(0, (sum + currentact*grid[i][j].memory)/(1+grid[i][j].memory))
		end
	end
end

-- Learning
function updateWeights()
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			-- Update the grid to new state
			grid[i][j].act = grid[i][j].new_act
			-- Update weights to minimize surprise, using the qudratic loss funtion
			local err = -2 * (grid[i][j].act - grid[i][j].past_act)
			grid[i][j].err = err
			for w = 1, N_WEIGHTS do
				grid[i][j].weights[w] = grid[i][j].weights[w] + grid[i][j].learningrate * err * math.tanh(grid[i][j].act)
			end
		end
	end
end

-- Reproduction
function updateReproduction()
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			local lower = (MIN_LEARNING / grid[i][j].learningrate + grid[i][j].memory/2)^2 / 64
			if grid[i][j].act < lower and grid[i][j].past_act < lower and grid[i][j].past2_act < lower then
				-- When dead, get new genes
				local nI = (i + math.random(-1, 1))
				local nJ = (j + math.random(-1, 1))
				nI, nJ = wrapAroundGrid(nI, nJ)
				if grid[nI][nJ].act > grid[nI][nJ].past_act then
					copyGenes(i, j, nI, nJ)
				end
			end
		end
	end
end

-- Mutation
function updateMutation()
	-- Pick several cells at random to mutate
	for n = 1, math.floor(N_MUTATIONS * GRID_WIDTH/100 * GRID_HEIGHT/100) do
		local i, j = math.random(1, GRID_WIDTH), math.random(1, GRID_HEIGHT)
		-- Introduce some noise to restart dead simulations and to select for more robust patterns
		grid[i][j].act = grid[i][j].act + math.random()/100
		-- Always change some wieght
		local w = math.random(1, N_WEIGHTS)
		grid[i][j].weights[w] = mutate(grid[i][j].weights[w])
		-- Modify some gene to mark changed cells
		local m = math.random(1,3)
		if m == 1 then
			grid[i][j].color1 = mutate(grid[i][j].color1)
		elseif m == 2 then
			grid[i][j].memory = mutate(grid[i][j].memory)
		elseif m == 3 then
			grid[i][j].learningrate = mutate(grid[i][j].learningrate)
		end
		-- Cells tend to prefer the lowest possible learningrate. Make sure it doesn't drop to zero.
		if grid[i][j].learningrate < MIN_LEARNING then
			grid[i][j].learningrate = MIN_LEARNING
		end
		if grid[i][j].memory > 1 then
			grid[i][j].memory = 1
		end
	end
end

-- Key events
function love.keypressed(key, scancode, isrepeat)
	if key == "space" then -- toggle pause simulation
		pause = not(pause)
	elseif key == "a" then -- continue paused simulation one time step
		step = true
	elseif key == "0" then
		prev_view_mode = view_mode
		view_mode = 0
	elseif key == "1" then
		prev_view_mode = view_mode
		view_mode = 1
	elseif key == "2" then
		prev_view_mode = view_mode
		view_mode = 2
	elseif key == "3" then
		prev_view_mode = view_mode
		view_mode = 3
	elseif key == "4" then
		prev_view_mode = view_mode
		view_mode = 4
	elseif key == "5" then
		prev_view_mode = view_mode
		view_mode = 5
	elseif key == "6" then
		prev_view_mode = view_mode
		view_mode = 6
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
						copyGenes(i, j, cellX, cellY)
						grid[i][j].act = grid[cellX][cellY].act
						grid[i][j].past_act = grid[cellX][cellY].past_act
					end
				end
			else
				-- Print information about the cell to terminal
				print("act = ",grid[cellX][cellY].act)
				print("err = ",grid[cellX][cellY].err)
				print("color1 = ",grid[cellX][cellY].color1)
				print("memory = ",grid[cellX][cellY].memory)
				print("learningrate = ",grid[cellX][cellY].learningrate)
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
		        red = math.tanh(grid[i][j].act - grid[i][j].act*grid[i][j].memory/8)
		        green = math.tanh(grid[i][j].past_act - grid[i][j].past_act*grid[i][j].color1/8)
		        blue = math.tanh(
					grid[i][j].act + grid[i][j].past_act + grid[i][j].past2_act/4
					+ (grid[i][j].act + grid[i][j].past_act)*grid[i][j].learningrate/8
				)
			elseif view_mode == 2 then -- error
		        red = math.tanh(grid[i][j].err)
		        green = math.tanh(grid[i][j].err)
		        blue = math.tanh(grid[i][j].err)
			elseif view_mode == 3 then -- focus learningrate
		        red = math.tanh(grid[i][j].past_act)/16
		        green = math.tanh(grid[i][j].act)/16
		        blue = math.tanh(grid[i][j].learningrate^0.5)
			elseif view_mode == 4 then -- focus color1
		        red = math.tanh(grid[i][j].color1)
		        green = math.tanh(grid[i][j].past_act)/16
		        blue = math.tanh(grid[i][j].act)/16
			elseif view_mode == 5 then -- focus memory
		        red = math.tanh(grid[i][j].act)/16
		        green = math.tanh(grid[i][j].memory)
		        blue = math.tanh(grid[i][j].past_act)/16
			elseif view_mode == 6 then -- focus memory
		        red = math.tanh(grid[i][j].color1)
		        green = math.tanh(grid[i][j].memory)
		        blue = math.tanh(grid[i][j].learningrate^0.5)
			end
            love.graphics.setColor(red, green, blue)
            love.graphics.rectangle("fill", x, y, CELLSIZE, CELLSIZE)
        end
    end
end

function love.quit()
	print("The simulation ran for "..generation.." generations. Seed was: "..seed..".")
end
