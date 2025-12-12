import sys
import json
import pygame

CELL_SIZE = 50
MARGIN = 50
INFO_PANEL_HEIGHT = 70

BG_COLOR = (0, 0, 0)
GRID_COLOR = (120, 120, 120)     
OBSTACLE_COLOR = (60, 60, 60)

AGENT_COLOR = (220, 0, 0)
AGENT_BORDER_COLOR = (0, 0, 0)
GOAL_TEXT_COLOR = (0, 0, 200)
PATH_COLOR = (0, 150, 255)
TEXT_COLOR = (255, 255, 255)

FONT_NAME = "ubuntu"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_agent_position(paths, agent_idx, t):
    path = paths[agent_idx]
    if t < len(path):
        return path[t]
    else:
        return path[-1]

def visualize(instance, solution):
    height = instance["height"]
    width = instance["width"]
    grid = instance["grid"]
    starts = instance["starts"]
    goals = instance["goals"]

    paths = solution["paths"]
    n_agents = len(paths)
    max_steps = max(len(p) for p in paths)

    pygame.init()
    pygame.display.set_caption("MAPF Visualizer")

    grid_pixel_width = width * CELL_SIZE
    grid_pixel_height = height * CELL_SIZE

    screen_width = grid_pixel_width + 2 * MARGIN
    screen_height = grid_pixel_height + 2 * MARGIN + INFO_PANEL_HEIGHT

    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(FONT_NAME, 24)
    small_font = pygame.font.SysFont(FONT_NAME, 20)

    current_t = 0  
    hovered_agent = None  

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    if current_t < max_steps - 1:
                        current_t += 1
                elif event.key == pygame.K_LEFT:
                    if current_t > 0:
                        current_t -= 1

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                col = (mx - MARGIN) // CELL_SIZE
                row = (my - MARGIN) // CELL_SIZE

                if 0 <= col < width and 0 <= row < height:
                    hovered_agent = None
                    for i in range(n_agents):
                        r, c = get_agent_position(paths, i, current_t)
                        if r == row and c == col:
                            hovered_agent = i
                            break
                else:
                    hovered_agent = None

        screen.fill(BG_COLOR)

        t_text = font.render(
            f"timestep: {current_t} / {max_steps - 1}", True, TEXT_COLOR
        )
        t_text_y = max(10, MARGIN - 30)
        screen.blit(t_text, (MARGIN, t_text_y))

        for r in range(height):
            for c in range(width):
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + r * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                # Nền ô
                if grid[r][c] == 1:
                    pygame.draw.rect(screen, OBSTACLE_COLOR, rect)
                else:
                    pygame.draw.rect(screen, (255, 255, 255), rect)

                pygame.draw.rect(screen, GRID_COLOR, rect, 2)

        for i, goal in enumerate(goals):
            gr, gc = goal  # [row, col]
            gx = MARGIN + gc * CELL_SIZE
            gy = MARGIN + gr * CELL_SIZE

            text = font.render(str(i + 1), True, GOAL_TEXT_COLOR)
            text_rect = text.get_rect(center=(gx + CELL_SIZE // 2,
                                              gy + CELL_SIZE // 2))
            screen.blit(text, text_rect)

        if hovered_agent is not None:
            path = paths[hovered_agent]

            if len(path) >= 2:
                points = []
                for (pr, pc) in path:
                    px = MARGIN + pc * CELL_SIZE + CELL_SIZE // 2
                    py = MARGIN + pr * CELL_SIZE + CELL_SIZE // 2
                    points.append((px, py))

                pygame.draw.lines(screen, PATH_COLOR, False, points, 3)

            for (pr, pc) in path:
                px = MARGIN + pc * CELL_SIZE + CELL_SIZE // 2
                py = MARGIN + pr * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.circle(screen, PATH_COLOR, (px, py), 4)


        for i in range(n_agents):
            r, c = get_agent_position(paths, i, current_t)
            x = MARGIN + c * CELL_SIZE
            y = MARGIN + r * CELL_SIZE
            rect = pygame.Rect(x + 4, y + 4, CELL_SIZE - 8, CELL_SIZE - 8)

            pygame.draw.rect(screen, AGENT_COLOR, rect)
            pygame.draw.rect(screen, AGENT_BORDER_COLOR, rect, 2)

            agent_label = small_font.render(str(i + 1), True, (255, 255, 255))
            label_rect = agent_label.get_rect(center=rect.center)
            screen.blit(agent_label, label_rect)


        info_y = MARGIN + grid_pixel_height + 10
        if hovered_agent is not None:
            hover_text = small_font.render(
                f"Agent: {hovered_agent + 1}", True, TEXT_COLOR
            )
            screen.blit(hover_text, (MARGIN, info_y))

        pygame.display.flip()

    pygame.quit()
