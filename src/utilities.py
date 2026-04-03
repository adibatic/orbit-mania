import pygame
import math

class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))

        # Create fonts
        self.large_font = pygame.font.Font(None, 64)
        self.small_font = pygame.font.Font(None, 40)

        # Load images
        self.background_image = pygame.image.load("assets/background.png")
        self.planet_image = pygame.image.load("assets/planet.png")
        self.player_image = pygame.image.load("assets/player.png")
        self.obstacle_image = pygame.image.load("assets/obstacle.png")
        self.explosion_image = pygame.image.load("assets/explosion.png")
        self.shield_image = pygame.image.load("assets/shield.png")
        self.energy_packet_image = pygame.image.load("assets/energy_packet.png")

        # Load background music
        pygame.mixer.init()
        self.background_music = pygame.mixer.music.load("assets/background_music.mp3")
        pygame.mixer.music.play(-1)
        
        # Load sounds
        self.activate_shield_sound = pygame.mixer.Sound("assets/activate_shield.mp3")
        self.activate_shield_sound.set_volume(20.0)
        self.energy_packet_sound = pygame.mixer.Sound("assets/energy_packet.mp3")
        self.energy_packet_sound.set_volume(20.0)
        self.game_over_sound = pygame.mixer.Sound("assets/game_over.mp3")
        self.game_over_sound.set_volume(3.0)

        # Application
        pygame.display.set_caption("Orbit Mania")

class Planet(pygame.sprite.Sprite):
    def __init__(self, image, center):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect(center=center)

    def draw_orbit(self, screen, radius, color):
        pygame.draw.circle(screen, color, self.rect.center, radius, width=1)

class Player(pygame.sprite.Sprite):
    def __init__(self, image, planet, shield_image, fuel_bar):
        super().__init__()
        self.image = image
        self.planet = planet
        self.rect = self.image.get_rect(center=(planet.rect.centerx, planet.rect.centery - 125))
        self.angle = 0
        self.radius = 125 # Instantaneous radius
        self.target_radius = 125  # Starting radius
        self.shield = Shield(shield_image, self)
        self.shield_active = False
        self.fuel_bar = fuel_bar

    def update(self):
        # Update player position based on angle and radius
        self.rect.centerx = self.planet.rect.centerx + self.radius * math.cos(math.radians(self.angle))
        self.rect.centery = self.planet.rect.centery + self.radius * math.sin(math.radians(self.angle))

        # Update angle for next frame
        self.angle += 0.8

        # Smooth transition between radii
        if self.radius != self.target_radius:
            if self.radius < self.target_radius:
                self.radius += 0.44
            else:
                self.radius -= 0.44
        
        self.shield.update()

        if self.shield_active:
            self.shield.activate()
        else:
            self.shield.deactivate()

    def change_radius(self, new_radius):
        self.target_radius = new_radius

    def activate_shield(self):
        self.shield_active = True

    def deactivate_shield(self):
        self.shield_active = False

class Shield(pygame.sprite.Sprite):
    def __init__(self, image, player):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect(center=player.rect.center)
        self.player = player
        self.active = False

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def update(self):
        self.rect.center = self.player.rect.center

class FuelBar:
    def __init__(self, world, max_fuel, x, y):
        self.world = world
        self.width = 200
        self.height = 20
        self.x = x
        self.y = y
        self.max_fuel = max_fuel
        self.current_fuel = max_fuel # Initialize with maximum fuel
        self.border_color = (255, 255, 255)
        self.fill_color = (255, 255, 0)

    def update(self, amount):
        self.current_fuel += amount
        if self.current_fuel < 0:
            self.current_fuel = 0
        elif self.current_fuel > self.max_fuel:
            self.current_fuel = self.max_fuel

class EnergyPacket(pygame.sprite.Sprite):
    def __init__(self, image, x, y, fuel_bar):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect(center=(x, y))
        self.fuel_bar = fuel_bar

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, image, x, y, speed, world):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = speed
        self.world = world

    def update(self):
        self.rect.move_ip(-self.speed, 0)
        if self.rect.right <= 0:
            self.rect.left = self.world.width  # Reset position when out of the screen
