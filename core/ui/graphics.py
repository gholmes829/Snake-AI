"""
Pygame wrapper allowing for easy, convenient, and customizable GUI windows.

Classes
-------
Engine
	Used to render window and blit shapes and text to screen.
"""

import pygame

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


class Engine:
	"""
	Used to render window and blit shapes and text to screen.

	Attributes
	----------
	targetFPS: int
		Target frames per second
	fontStyle: str
		Font style
	clock: pygame.time.Clock
		Used to regulate game ticks and FPS
	dt: float
		Delta time, measurement of latency between frames, used to achieve frame rate motion independence
	running: bool
		Whether engine is running
	fontCache: dict
		Caches rendered fonts, improves performance
	surfaceCache: dict
		Caches rendered surfaces, improves performance
	background: pygame.Surface
		Background for active window
	screen: pygame.Surface
		Background for entire screen
	offset: tuple
		(x, y) offset for position of active window relative to screen

	Public Methods
	--------------
	shouldRun() -> None:
		Determines if engine should keep running.
	clearScreen() -> None:
		Removes everything blitted on screen by covering everything with background.
	updateScreen() -> None:
		Renders necessary components to screen.
	exit() -> None:
		Has engine exit.
	scaleUp(coord: tuple) -> tuple:
		Scales (x, y) coords to fit resolution of screen.
	renderScene(func: callable, *args) -> None:
		Renders custom scene defined outside of this class in the form of customScene(engine: graphics.Engine...
	printToScreen(text: str, pos: tuple, fontSize: int, textColor: tuple, backgroundColor: tuple = None) -> None:
		Blits text to screen.
	renderRect(pos: tuple, size: tuple, fillColor: tuple, alpha: int = 255) -> None:
		Blits rect to screen.
	renderCircle(pos: tuple, radius: float, fillColor: tuple, alpha: int = 255) -> None:
		Blits circle to screen.
	renderLine(start: tuple, end: tuple, width: int, fillColor: tuple) -> None:
		Blits line to screen.
	"""
	colors = {
		"green": (0, 255, 0),
		"red": (255, 0, 0),
		"purple": (200, 100, 200),
		"blue": (0, 0, 255),
		"mediumBlue": (150, 150, 255),
		"lightBlue": (175, 175, 255),
		"black": (0, 0, 0),
		"white": (255, 255, 255),
	}

	def __init__(self,
				 screenSize: tuple,
				 numGrids: tuple,
				 checkered: bool = False,
				 targetFPS: int = 60,
				 title: str = "Untitled Game",
				 fontStyle: str = "impact",
				 gridColors: tuple = ("black", "white", "black"),
				 record = False,
				 ) -> None:
		"""
		Initializes engine, calculates aspect ratio and fits active window to screen.

		Parameters
		----------
		screenSize: tuple
			Resolution of GUI window
		numGrids: tuple
			Number of grid for active game window
		checkered: bool
			If active game window should have checkered background
		targetFPS: int
			Target frames per second
		title: str
			Window title
		fontStyle: str
			Font style
		gridColors: tuple
			Colors of grid if checkered in form (checker color 1, checker color 2, border color)
		"""
		self.targetFPS = targetFPS
		self.fontStyle = fontStyle

		pygame.init()
		pygame.font.init()

		self.clock = pygame.time.Clock()
		self.dt = None
		self.running = True
		pygame.display.set_caption(title)
		
		self.record = record
		self.t = 0        

		self.fontCache = {}
		self.surfaceCache = {}

		# aspect ratios
		screenAR, gridsAR = screenSize[0]/screenSize[1], numGrids[0]/numGrids[1]

		# position active window to fit on screen
		if gridsAR == screenAR:
			self.offset = (0, 0)
			backgroundSize = screenSize
		elif gridsAR > screenAR:
			compression = screenSize[0]/numGrids[0]
			backgroundSize = (numGrids[0]*compression, numGrids[1]*compression)
			self.offset = (0, 0.5*(screenSize[1] - backgroundSize[1]))
		else:
			compression = screenSize[1] / numGrids[1]
			backgroundSize = (numGrids[0] * compression, numGrids[1] * compression)
			self.offset = (0.5 * (screenSize[0] - backgroundSize[0]), 0)

		self.screen = pygame.display.set_mode(screenSize)
		self.surfaceCache[backgroundSize] = pygame.Surface(backgroundSize)
		self.background = self.surfaceCache[backgroundSize]

		self.gridSize = tuple([int(backgroundSize[0] / numGrids[0]), int(backgroundSize[1] / numGrids[1])])
		self.paddedGridSize = (self.gridSize[0] + 1, self.gridSize[1] + 1)

		# calculate clipping due to discrepancy of integer rounding
		clipping = (0.5*(backgroundSize[0] % numGrids[0]), 0.5 * (backgroundSize[1] % numGrids[1]))
		self.offset = (self.offset[0]+clipping[0], self.offset[1]+clipping[1])
		self.screen.set_clip(pygame.Rect(*self.offset, backgroundSize[0]-clipping[0]*2, backgroundSize[1]-clipping[1]*2))

		# create checkered background
		if checkered:
			for coord, val in Engine.checkerboard(numGrids).items():
				rect = pygame.Surface(self.gridSize)
				rect.fill(Engine.colors[gridColors[val]])
				self.background.blit(rect, (coord[0] * self.gridSize[0], coord[1] * self.gridSize[1]))

	def shouldRun(self) -> bool:
		"""
		Determines if engine should keep running.

		Returns
		-------
		bool: if engine is now running after checks
		"""
		if not self.running:
			return False
		self._handleEvents()
		self.dt = self.clock.tick(self.targetFPS) / 1000 * self.targetFPS
		return self.running

	def clearScreen(self) -> None:
		"""Removes everything blitted on screen by covering everything with background."""
		self.screen.blit(self.background, self.offset)

	def updateScreen(self) -> None:
		"""Renders necessary components to screen."""
		pygame.display.flip()
		if self.record:
			pygame.image.save(self.screen, "frames/screen_" + str(self.t) + ".png")
		self.t += 1

	def exit(self) -> None:
		"""Has engine exit."""
		if self.running:
			pygame.quit()

	def scaleUp(self, coord: tuple) -> tuple:
		"""
		Scales (x, y) coords to fit resolution of screen.

		Parameters
		----------
		coord: tuple
			(x, y) coord to scale up

		Returns
		-------
		tuple: scaled coord
		"""
		return coord[0] * self.gridSize[0] + self.offset[0], coord[1] * self.gridSize[1] + self.offset[1]

	def renderScene(self, func: callable, *args) -> None:
		"""
		Renders custom scene defined outside of this class in the form of customScene(engine: graphics.Engine...

		Parameters
		----------
		func: callable
			Function describing how to render custom scene
		*args
			Arguments to pass into func
		"""
		func(self, *args)

	def printToScreen(self, text: str, pos: tuple, fontSize: int, textColor: tuple, backgroundColor: tuple = None) -> None:
		"""
		Blits text to screen.

		Parameters
		----------
		text: str
			Text to display
		pos: tuple
			(x, y) pos on screen to display text
		fontSize: int
			Size of font
		textColor: tuple
			RGB color value
		backgroundColor: tuple, optional
			RGB color value for rect behind text
		"""
		if fontSize not in self.fontCache:
			self.fontCache[fontSize] = pygame.font.SysFont(self.fontStyle, fontSize)

		font = self.fontCache[fontSize]
		paddedOutput = " " + text + " "

		if backgroundColor is not None:
			text = font.render(paddedOutput, True, textColor, backgroundColor)
		else:
			text = font.render(paddedOutput, True, textColor)

		textRect = text.get_rect()
		textRect.center = pos
		self.screen.blit(text, textRect)

	def renderRect(self, pos: tuple, size: tuple, fillColor: tuple, alpha: int = 255) -> None:
		"""
		Blits rect to screen.

		Parameters
		----------
		pos: tuple
			(x, y) pos to blit rect to screen
		size: tuple
			(x, y) size of rect
		fillColor: tuple
			RGB values for color of rect
		alpha: int, default=255
			Transparency value (0-255) of rect
		"""
		if size not in self.surfaceCache:
			self.surfaceCache[size] = pygame.Surface(size)

		surface = self.surfaceCache[size]
		surface.set_alpha(alpha)
		surface.fill(fillColor)
		self.screen.blit(surface, pos)

	def renderCircle(self, pos: tuple, radius: float, fillColor: tuple, alpha: int = 255) -> None:
		"""
		Blits circle to screen.

		Parameters
		----------
		pos: tuple
			(x, y) pos to blit rect to screen
		radius: float
			Radius of circle
		fillColor: tuple
			RGB values for color of rect
		alpha: int, default=255
			Transparency value (0-255) of rect
		"""
		frameSize = (radius * 2, radius * 2)
		rel_x = radius
		rel_y = radius

		if frameSize not in self.surfaceCache:
			self.surfaceCache[frameSize] = pygame.Surface(frameSize)

		surface = self.surfaceCache[frameSize]
		surface.fill(Engine.colors["white"])
		surface.set_colorkey(Engine.colors["white"])
		surface.set_alpha(alpha)

		pygame.draw.circle(surface, fillColor, (rel_x, rel_y), radius)
		self.screen.blit(surface, pos)

	def renderLine(self, start: tuple, end: tuple, width: int, fillColor: tuple) -> None:
		"""
		Blits line to screen.

		Parameters
		----------
		start: tuple
			(x, y) pos for start point of line
		end: tuple
			(x, y) pos for end point of line
		width: int
			Value denoting width of line
		fillColor: tuple
			RGB values for color of rect
		"""
		pygame.draw.line(self.screen, fillColor, start, end, width)

	def _handleEvents(self) -> None:
		"""Handles events from Pygame's event queue. pygame.QUIT occurs when "X" on top right corner is clicked."""
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False
				pygame.quit()

	@staticmethod
	def checkerboard(n: tuple, border: bool = True) -> dict:
		"""
		Creates checkerboard representation.
		x x x x x . . .
		x o x o x
		x x o x o
		.		. . .
		.		.
		.		.

		Parameters
		----------
		n: tuple
			Size of checkerboard
		border: bool
			Whether border should be included on edges

		Returns
		-------
		dict: Keys are (x, y) coords, values are 0, 1, or 2
		"""
		board = {}
		edges = ({0, n[0] - 1}, {0, n[1] - 1})
		for i in range(n[0]):
			for j in range(n[1]):
				if border and any([(i, j)[k] in edges[k] for k in range(2)]):
					board[(i, j)] = 2
				elif (i + j) % 2 == 0:
					board[(i, j)] = 1
				else:
					board[(i, j)] = 0
		return board
