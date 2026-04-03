import pygame
import time
import argparse
import numpy as np
import os
import sys

# Append current dir to path so env/utilities can be imported from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import OrbitEnv
import utilities

class AppMain:
    def __init__(self, ai_mode=False, model_path=None):
        pygame.init()
        self.world = utilities.World(800, 600)
        self.env = OrbitEnv()
        self.obs, _ = self.env.reset()
        self.stopwatch = pygame.time.Clock()
        self.start_time = time.time()
        self.last_fuel = self.env.fuel
        self.shield_sound_played = False

        self.ai_mode = ai_mode
        self.model = None
        if self.ai_mode:
            from stable_baselines3 import PPO
            print(f"Loading model from {model_path}...")
            try:
                self.model = PPO.load(model_path)
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.ai_mode = False

    def format_time(self, elapsed_time):
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        return "{:02}:{:02}".format(minutes, seconds)

    def draw_centered(self, image, pos_math):
        """Convert math coords (planet=origin) to Pygame screen coords."""
        x = pos_math[0] + 400
        y = pos_math[1] + 300
        rect = image.get_rect(center=(int(x), int(y)))
        self.world.screen.blit(image, rect)

    def run(self):
        running = True
        while running:
            elapsed_time = time.time() - self.start_time

            switch_input = 0
            shield_input = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # Space fires once per press — triggers smooth orbit transition
                    if not self.ai_mode and event.key == pygame.K_SPACE:
                        switch_input = 1

            if not self.ai_mode:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_s]:
                    shield_input = 1
                action = [switch_input, shield_input]
            else:
                action, _states = self.model.predict(self.obs, deterministic=True)

            self.obs, reward, terminated, truncated, info = self.env.step(action)

            # Sound: energy packet collected
            if self.env.fuel - self.last_fuel > 15:
                self.world.energy_packet_sound.play()
            self.last_fuel = self.env.fuel

            if terminated:
                # Explosion then game-over screen
                explosion_rect = self.world.explosion_image.get_rect(
                    center=(int(self.env.player_pos[0]) + 400,
                            int(self.env.player_pos[1]) + 300)
                )
                self.world.screen.blit(self.world.explosion_image, explosion_rect)
                pygame.display.flip()
                pygame.mixer.music.stop()
                self.world.game_over_sound.play()
                pygame.time.wait(500)

                self.world.screen.blit(self.world.background_image, (0, 0))
                go_text = self.world.large_font.render("GAME OVER", True, (255, 255, 255))
                self.world.screen.blit(go_text, go_text.get_rect(center=(400, 280)))
                lt_text = self.world.small_font.render(
                    "Time: " + self.format_time(elapsed_time), True, (255, 255, 255)
                )
                self.world.screen.blit(lt_text, lt_text.get_rect(center=(400, 340)))
                pygame.display.flip()
                pygame.time.wait(3500)
                break

            # ── Render ──────────────────────────────────────────────────────
            self.world.screen.blit(self.world.background_image, (0, 0))

            # Orbit guide rings
            pygame.draw.circle(self.world.screen, (255, 255, 255), (400, 300), 125, width=1)
            pygame.draw.circle(self.world.screen, (255, 255, 255), (400, 300), 225, width=1)

            # Planet
            self.draw_centered(self.world.planet_image, [0, 0])

            # Energy packets
            for ep in self.env.energy_packets:
                self.draw_centered(self.world.energy_packet_image, ep['pos'])

            # Obstacles
            for ob in self.env.obstacles:
                self.draw_centered(self.world.obstacle_image, ob['pos'])

            # Shield (drawn behind player)
            if self.env.shield_active:
                self.draw_centered(self.world.shield_image, self.env.player_pos)
                if not self.shield_sound_played:
                    self.world.activate_shield_sound.play()
                    self.shield_sound_played = True
            else:
                self.shield_sound_played = False

            # Player
            self.draw_centered(self.world.player_image, self.env.player_pos)

            # ── HUD ─────────────────────────────────────────────────────────
            fuel_ratio = self.env.fuel / self.env.MAX_FUEL
            fuel_label = self.world.small_font.render("Fuel", True, (255, 255, 255))
            self.world.screen.blit(fuel_label, (20, 10))
            pygame.draw.rect(self.world.screen, (255, 255, 255), (20, 40, 200, 20), 3)
            pygame.draw.rect(self.world.screen, (255, 255, 0), (20, 40, int(200 * fuel_ratio), 20))

            orbit_label = self.world.small_font.render(
                f"Orbit: {self.env.radius:.0f} px", True, (0, 200, 255)
            )
            self.world.screen.blit(orbit_label, (20, 70))

            time_text = self.world.large_font.render(
                self.format_time(elapsed_time), True, (255, 255, 0)
            )
            self.world.screen.blit(time_text, time_text.get_rect(topright=(790, 10)))

            mode_label = self.world.small_font.render(
                "AI MODE" if self.ai_mode else "SPACE: switch orbit  |  S: shield",
                True, (180, 180, 180)
            )
            self.world.screen.blit(mode_label, mode_label.get_rect(midbottom=(400, 595)))

            pygame.display.flip()
            self.stopwatch.tick(60)

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", action="store_true", help="Run AI agent")
    parser.add_argument("--model", type=str, default="models/ppo_orbit_mania.zip")
    args = parser.parse_args()
    app = AppMain(ai_mode=args.ai, model_path=args.model)
    app.run()
