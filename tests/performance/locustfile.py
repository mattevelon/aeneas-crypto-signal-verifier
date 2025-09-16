"""
Load testing configuration for AENEAS API.
Uses Locust for simulating user traffic and stress testing.
"""

from locust import HttpUser, TaskSet, task, between, constant
import random
import json
import time
from datetime import datetime, timedelta


class UserBehavior(TaskSet):
    """Simulates typical user behavior."""
    
    def on_start(self):
        """Called when a user starts."""
        # Login to get auth token
        self.login()
        
    def login(self):
        """Authenticate and get token."""
        response = self.client.post("/api/v1/auth/login", 
            data={"username": "testuser", "password": "testpass123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}
            
    @task(10)
    def get_health(self):
        """Check health endpoint."""
        self.client.get("/api/v1/health")
        
    @task(5)
    def get_signals(self):
        """Get signals list."""
        self.client.get("/api/v1/signals", 
                       headers=self.headers,
                       params={"limit": 20, "offset": 0})
        
    @task(3)
    def get_signal_detail(self):
        """Get specific signal details."""
        # Simulate getting a signal ID
        signal_id = f"550e8400-e29b-41d4-a716-{random.randint(100000000000, 999999999999)}"
        self.client.get(f"/api/v1/signals/{signal_id}",
                       headers=self.headers,
                       catch_response=True)
        
    @task(2)
    def get_statistics(self):
        """Get system statistics."""
        endpoints = [
            "/api/v1/stats/overview",
            "/api/v1/stats/channels",
            "/api/v1/stats/performance",
            "/api/v1/stats/risk-metrics"
        ]
        
        endpoint = random.choice(endpoints)
        self.client.get(endpoint, headers=self.headers)
        
    @task(1)
    def submit_feedback(self):
        """Submit signal feedback."""
        feedback = {
            "signal_id": f"550e8400-e29b-41d4-a716-{random.randint(100000000000, 999999999999)}",
            "rating": random.randint(1, 5),
            "comment": "Test feedback from load testing",
            "outcome": random.choice(["success", "partial", "failure"])
        }
        
        self.client.post("/api/v1/feedback",
                        json=feedback,
                        headers=self.headers,
                        catch_response=True)


class WebSocketUser(TaskSet):
    """Simulates WebSocket connections."""
    
    def on_start(self):
        """Connect to WebSocket."""
        self.ws_url = "/ws/v2/connect"
        self.connect_websocket()
        
    def connect_websocket(self):
        """Establish WebSocket connection."""
        # Note: Locust doesn't natively support WebSocket
        # This simulates the HTTP upgrade request
        self.client.get(self.ws_url,
                       headers={"Upgrade": "websocket",
                               "Connection": "Upgrade"})
        
    @task(5)
    def send_heartbeat(self):
        """Send WebSocket heartbeat."""
        # Simulate heartbeat
        time.sleep(0.1)
        
    @task(2)
    def subscribe_to_signals(self):
        """Subscribe to signal updates."""
        # Simulate subscription
        time.sleep(0.1)
        
    @task(1)
    def unsubscribe(self):
        """Unsubscribe from updates."""
        # Simulate unsubscription
        time.sleep(0.1)


class APIUser(HttpUser):
    """Standard API user."""
    tasks = [UserBehavior]
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when user starts."""
        print(f"User starting at {datetime.now()}")
        
    def on_stop(self):
        """Called when user stops."""
        print(f"User stopping at {datetime.now()}")


class PowerUser(HttpUser):
    """Power user with higher request rate."""
    tasks = [UserBehavior]
    wait_time = constant(0.5)


class WebSocketOnlyUser(HttpUser):
    """User that only uses WebSocket."""
    tasks = [WebSocketUser]
    wait_time = between(5, 10)


class StressTestUser(HttpUser):
    """Stress test user for peak load simulation."""
    
    class StressBehavior(TaskSet):
        """Aggressive behavior for stress testing."""
        
        @task
        def hammer_endpoint(self):
            """Rapidly hit endpoints."""
            endpoints = [
                "/api/v1/health",
                "/api/v1/signals",
                "/api/v1/stats/overview"
            ]
            
            for _ in range(10):
                endpoint = random.choice(endpoints)
                self.client.get(endpoint, catch_response=True)
                
    tasks = [StressBehavior]
    wait_time = constant(0.1)


# Custom load shape for realistic traffic patterns
class RealisticLoadShape:
    """Simulates realistic daily traffic patterns."""
    
    def tick(self):
        """Generate user count based on time of day."""
        current_hour = datetime.now().hour
        
        # Simulate daily pattern
        if 0 <= current_hour < 6:
            # Night - low traffic
            return 10, 1
        elif 6 <= current_hour < 9:
            # Morning ramp-up
            return 50, 5
        elif 9 <= current_hour < 12:
            # Morning peak
            return 100, 10
        elif 12 <= current_hour < 14:
            # Lunch dip
            return 70, 5
        elif 14 <= current_hour < 18:
            # Afternoon peak
            return 120, 10
        elif 18 <= current_hour < 22:
            # Evening moderate
            return 60, 5
        else:
            # Late evening
            return 20, 2


# Performance test scenarios
class PerformanceTestScenarios:
    """Different performance test scenarios."""
    
    @staticmethod
    def smoke_test():
        """Light load to verify system works."""
        return {
            "users": 5,
            "spawn_rate": 1,
            "duration": 60  # 1 minute
        }
        
    @staticmethod
    def load_test():
        """Normal expected load."""
        return {
            "users": 100,
            "spawn_rate": 10,
            "duration": 600  # 10 minutes
        }
        
    @staticmethod
    def stress_test():
        """Beyond normal capacity."""
        return {
            "users": 500,
            "spawn_rate": 50,
            "duration": 300  # 5 minutes
        }
        
    @staticmethod
    def spike_test():
        """Sudden traffic spike."""
        return {
            "users": 1000,
            "spawn_rate": 100,
            "duration": 120  # 2 minutes
        }
        
    @staticmethod
    def soak_test():
        """Extended duration test."""
        return {
            "users": 200,
            "spawn_rate": 10,
            "duration": 3600  # 1 hour
        }
