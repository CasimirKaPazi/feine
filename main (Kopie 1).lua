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
	CELLSIZE = 2 -- pixel on screen
	-- affecting cells
	RANGE = 2 -- how far from origin weights are drawn
	N_WEIGHTS = (RANGE * 2 + 1)^2 - 1 -- number of connections increases exponentially with range
	MIN_LEARNING = 0.0005 -- found through trial and error. Affects the ratio between learning and evolution.
	N_MUTATIONS = 2 -- mutations per 100x100 area per time step
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
			grid[i][j].new_act = math.max(0, (sum*3 + currentact*grid[i][j].memory)/(1*3+grid[i][j].memory))
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
			local learningrate = (grid[i][j].learning^4 + MIN_LEARNING)
			grid[i][j].err = err
			for w = 1, N_WEIGHTS do
				for z = 1, 5 do
				grid[i][j][z].weights[w] = grid[i][j][z].weights[w] + learningrate * err * math.tanh(grid[i][j].act)
				end
			end
		end
	end
end

-- Reproduction
function updateReproduction()
	for i = 1, GRID_WIDTH do
		for j = 1, GRID_HEIGHT do
			local lower = ( math.tanh(1/grid[i][j].learning) * math.tanh(grid[i][j].memory+0.001) )^0.5 /2
			if math.abs(grid[i][j].err) + grid[i][j].act/2 < lower then
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
		grid[i][j].act = grid[i][j].act + math.random()/50
		-- Modify genes
		local m = math.random(1,10)
		if m == 1 then -- color1
			grid[i][j].color1 = mutate(grid[i][j].color1)
		elseif m == 2 then -- memory
			grid[i][j].memory = mutate(grid[i][j].memory)
		elseif m == 3 then -- leanring rate
			grid[i][j].learning = mutate(grid[i][j].learning)
		elseif m == 4 then -- swap color1 and memory
			local swap = grid[i][j].color1
			grid[i][j].color1 = grid[i][j].memory
			grid[i][j].memory = swap
		elseif m == 5 then -- swap color1 and memory
			local swap = grid[i][j].memory
			grid[i][j].memory = grid[i][j].learning
			grid[i][j].learning = swap
		elseif m == 6 then -- swap color1 and memory
			local swap = grid[i][j].learning
			grid[i][j].learning = grid[i][j].color1
			grid[i][j].color1 = swap
		elseif m == 7 then -- invert random weight
			local w = math.random(1, N_WEIGHTS)
			local z = math.random(1, 5)
			grid[i][j][z].weights[w] = mutate(grid[i][j][z].weights[w]+0.5)-0.5
		elseif m == 8 then -- copy one weight onto another
			local w = math.random(1, N_WEIGHTS)
			local k = math.random(1, N_WEIGHTS)
			local z = math.random(1, 5)
			grid[i][j][z].weights[w] = grid[i][j][z].weights[k]
		elseif m == 9 then -- invert random weight
			local w = math.random(1, N_WEIGHTS)
			local z = math.random(1, 5)
			grid[i][j][z].weights[w] = -1* grid[i][j][z].weights[w]
		elseif m == 10 then -- merge genes with direct neighbor
			local x = math.random(-1,1)
			local y = math.random(-1,1)
			if math.abs(x) + math.abs(y) == 1 then
				local nI, nJ = wrapAroundGrid(i + math.random(-1,1), j + math.random(-1,1))
				mixGenes(i, j, nI, nJ)
			end
		end
		if grid[i][j].learning < 0 then
			grid[i][j].learning = 0
		end
		if grid[i][j].memory > 1 then
			grid[i][j].memory = 1
		end
		if grid[i][j].memory < 0 then
			grid[i][j].memory = 0
		end
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
						grid[i][j].act = math.random()
						grid[i][j].past_act = math.random()
						grid[i][j].past2_act = math.random()
					end
				end
			else
				-- Print information about the cell to terminal
				print("act = ",grid[cellX][cellY].act)
				print("err = ",grid[cellX][cellY].err)
				print("color1 = ",grid[cellX][cellY].color1)
				print("memory = ",grid[cellX][cellY].memory)
				print("learning = ",grid[cellX][cellY].learning)
				for w = 1, N_WEIGHTS do
					for z = 1, 5 do
						print("weight "..w.." = ",grid[cellX][cellY][].weights[w])
					end
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
		        red = math.tanh(grid[i][j].act*2 - grid[i][j].err/8)
		        green = math.tanh(grid[i][j].past_act*2 - grid[i][j].err/8)
		        blue = math.tanh(grid[i][j].act*2 + grid[i][j].past_act*2 + grid[i][j].past2_act/2)
			elseif view_mode == 2 then -- error
		        red = math.tanh(grid[i][j].err)
		        green = math.tanh(grid[i][j].err)
		        blue = math.tanh(grid[i][j].err)
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
			end
            love.graphics.setColor(red, green, blue)
            love.graphics.rectangle("fill", x, y, CELLSIZE, CELLSIZE)
        end
    end
end

function love.quit()
	print("The simulation ran for "..generation.." generations. Seed was: "..seed..".")
end
