import pygame

COLOR_TEXT_MUTED = (160, 160, 180)
COLOR_START = (6, 214, 160)
COLOR_AGENT = (255, 209, 102)

class Button:
    def __init__(self, x, y, w, h, text, color, hover_color, text_color=(255, 255, 255), action=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.action = action
        self.is_hovered = False

    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (255, 255, 255, 40), self.rect, width=1, border_radius=8)

        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered and self.action:
                self.action()
                return True
        return False


class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, start_val, label, is_float=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = start_val
        self.label = label
        self.is_float = is_float
        self.grabbed = False

        self.handle_w = 14
        self.handle_rect = pygame.Rect(0, 0, self.handle_w, h + 8)
        self.update_handle_pos()

    def update_handle_pos(self):
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        self.handle_rect.centerx = self.rect.x + int(ratio * self.rect.width)
        self.handle_rect.centery = self.rect.centery

    def draw(self, screen, font):
        val_text = f"{self.val:.3f}" if self.is_float else f"{int(self.val)}"
        label_surf = font.render(f"{self.label}: {val_text}", True, COLOR_TEXT_MUTED)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 22))

        # 트랙
        pygame.draw.rect(screen, (60, 60, 75), self.rect, border_radius=4)
        # 채워진 부분
        filled_w = self.handle_rect.centerx - self.rect.x
        if filled_w > 0:
            filled_rect = pygame.Rect(self.rect.x, self.rect.y, filled_w, self.rect.height)
            pygame.draw.rect(screen, COLOR_START, filled_rect, border_radius=4)

        # 핸들
        color = COLOR_AGENT if self.grabbed else (240, 240, 245)
        pygame.draw.rect(screen, color, self.handle_rect, border_radius=4)

    def handle_event(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect.collidepoint(mouse_pos) or self.rect.collidepoint(mouse_pos):
                self.grabbed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.grabbed = False

        if self.grabbed and event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            rel_x = max(0, min(mouse_pos[0] - self.rect.x, self.rect.width))
            ratio = rel_x / self.rect.width
            self.val = self.min_val + ratio * (self.max_val - self.min_val)
            if not self.is_float:
                self.val = int(round(self.val))
            self.update_handle_pos()
            return True
        return False


class ToggleButton(Button):
    def __init__(self, x, y, w, h, text, color, hover_color, active_color, active_hover_color, is_active=False, text_color=(255, 255, 255), action=None):
        super().__init__(x, y, w, h, text, color, hover_color, text_color, action)
        self.active_color = active_color
        self.active_hover_color = active_hover_color
        self.is_active = is_active

    def draw(self, screen, font):
        if self.is_active:
            color = self.active_hover_color if self.is_hovered else self.active_color
        else:
            color = self.hover_color if self.is_hovered else self.color
        
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        
        # 활성화 상태면 밝은 흰색 테두리로 강조, 비활성화면 옅은 테두리
        border_color = (255, 255, 255) if self.is_active else (255, 255, 255, 40)
        border_width = 2 if self.is_active else 1
        pygame.draw.rect(screen, border_color, self.rect, width=border_width, border_radius=8)

        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

