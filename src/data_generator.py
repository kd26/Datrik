import sqlite3
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import os
import csv
from typing import List, Dict, Tuple

class FoodDeliveryDataGenerator:
    def __init__(self, seed: int = 42, days: int = 90):
        self.seed = seed
        self.days = days
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
        
        # Configuration
        self.config = {
            'users': 15000,
            'restaurants': 1200,
            'couriers': 600,
            'avg_orders_per_day': 500,
            'avg_items_per_order': 2.1,
            'sessions_multiplier': 2.2,  # sessions per order
        }
        
        # Date range
        self.end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
        self.start_date = self.end_date - timedelta(days=days-1)
        
        # Cache for generated data
        self.users_data = []
        self.restaurants_data = []
        self.couriers_data = []
        self.orders_data = []
        self.order_items_data = []
        self.sessions_data = []
        self.marketing_events_data = []
        self.reviews_data = []
        
    def generate_all_data(self):
        """Generate all tables' data"""
        print("Generating users...")
        self.generate_users()
        
        print("Generating restaurants...")
        self.generate_restaurants()
        
        print("Generating couriers...")
        self.generate_couriers()
        
        print("Generating orders and order items...")
        self.generate_orders_and_items()
        
        print("Generating sessions...")
        self.generate_sessions()
        
        print("Generating marketing events...")
        self.generate_marketing_events()
        
        print("Generating reviews...")
        self.generate_reviews()
        
        print("Data generation complete!")
        
    def generate_users(self):
        """Generate users data with realistic demographics"""
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
                 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
                 'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis',
                 'Seattle', 'Denver', 'Washington']
        
        cuisines = ['Italian', 'Chinese', 'Mexican', 'Indian', 'Thai', 'Japanese', 'American',
                   'Mediterranean', 'Korean', 'Vietnamese', 'Greek', 'French']
        
        for i in range(self.config['users']):
            registration_date = self.fake.date_time_between(
                start_date='-2y', end_date=self.start_date
            )
            
            city = random.choice(cities)
            
            user = {
                'user_id': i + 1,
                'email': self.fake.unique.email(),
                'phone': self.fake.phone_number()[:15],
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'date_of_birth': self.fake.date_of_birth(minimum_age=18, maximum_age=70),
                'gender': random.choice(['male', 'female', 'other']),
                'registration_date': registration_date,
                'city': city,
                'postal_code': self.fake.zipcode(),
                'latitude': round(self.fake.latitude(), 6),
                'longitude': round(self.fake.longitude(), 6),
                'is_active': random.choices([1, 0], weights=[0.85, 0.15])[0],
                'total_orders': 0,  # Will be updated after generating orders
                'lifetime_value': 0.0,  # Will be updated after generating orders
                'preferred_cuisine': random.choice(cuisines)
            }
            
            self.users_data.append(user)
    
    def generate_restaurants(self):
        """Generate restaurants data"""
        cuisine_types = ['Italian', 'Chinese', 'Mexican', 'Indian', 'Thai', 'Japanese', 'American',
                        'Mediterranean', 'Korean', 'Vietnamese', 'Greek', 'French', 'Pizza', 
                        'Burger', 'Sushi', 'BBQ', 'Seafood', 'Vegetarian', 'Fast Food', 'Breakfast']
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia']
        
        restaurant_names = {
            'Italian': ['Mario\'s Kitchen', 'Bella Vita', 'Casa Roma', 'Giuseppe\'s', 'Amore Italiano'],
            'Chinese': ['Golden Dragon', 'Panda Garden', 'Lucky Bamboo', 'Jade Palace', 'Great Wall'],
            'Mexican': ['El Mariachi', 'Taco Libre', 'Casa Miguel', 'Fiesta Cantina', 'Los Amigos'],
            'Indian': ['Spice Palace', 'Taj Mahal', 'Curry House', 'Mumbai Express', 'Delhi Garden'],
            'Thai': ['Thai Smile', 'Bangkok Kitchen', 'Spicy Basil', 'Golden Buddha', 'Thai Paradise'],
            'Japanese': ['Sakura Sushi', 'Tokyo Express', 'Ramen House', 'Zen Garden', 'Sushi Master'],
            'American': ['All American Diner', 'Liberty Grill', 'Stars & Stripes', 'Hometown Kitchen', 'Classic Cafe'],
            'Pizza': ['Tony\'s Pizza', 'Slice Heaven', 'Mama Mia\'s', 'Perfect Pie', 'Pizza Corner']
        }
        
        for i in range(self.config['restaurants']):
            cuisine = random.choice(cuisine_types)
            base_names = restaurant_names.get(cuisine, ['Restaurant', 'Kitchen', 'Cafe', 'Bistro', 'Eatery'])
            name = f"{random.choice(base_names)} {random.randint(1, 999)}"
            
            registration_date = self.fake.date_time_between(
                start_date='-3y', end_date=self.start_date
            )
            
            restaurant = {
                'restaurant_id': i + 1,
                'name': name,
                'cuisine_type': cuisine,
                'rating': round(random.uniform(3.0, 5.0), 1),
                'total_reviews': random.randint(50, 2000),
                'address': self.fake.street_address(),
                'city': random.choice(cities),
                'postal_code': self.fake.zipcode(),
                'latitude': round(self.fake.latitude(), 6),
                'longitude': round(self.fake.longitude(), 6),
                'phone': self.fake.phone_number()[:15],
                'opening_time': f"{random.randint(6, 10)}:00",
                'closing_time': f"{random.randint(21, 24)}:00",
                'delivery_fee': round(random.uniform(1.99, 4.99), 2),
                'minimum_order': round(random.uniform(10.0, 25.0), 2),
                'average_prep_time': random.randint(15, 45),
                'is_active': random.choices([1, 0], weights=[0.9, 0.1])[0],
                'registration_date': registration_date
            }
            
            self.restaurants_data.append(restaurant)
    
    def generate_couriers(self):
        """Generate couriers data"""
        vehicle_types = ['bike', 'scooter', 'car', 'walking']
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia']
        
        for i in range(self.config['couriers']):
            registration_date = self.fake.date_time_between(
                start_date='-1y', end_date=self.start_date
            )
            
            courier = {
                'courier_id': i + 1,
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'phone': self.fake.phone_number()[:15],
                'email': self.fake.email(),
                'vehicle_type': random.choice(vehicle_types),
                'rating': round(random.uniform(4.0, 5.0), 1),
                'total_deliveries': 0,  # Will be updated after generating orders
                'registration_date': registration_date,
                'is_active': random.choices([1, 0], weights=[0.8, 0.2])[0],
                'current_city': random.choice(cities)
            }
            
            self.couriers_data.append(courier)
    
    def generate_orders_and_items(self):
        """Generate orders and order items with realistic patterns"""
        total_orders = self.config['avg_orders_per_day'] * self.days
        
        # Day of week seasonality (Monday = 0, Sunday = 6)
        day_multipliers = {0: 0.8, 1: 0.9, 2: 0.85, 3: 0.95, 4: 1.3, 5: 1.4, 6: 1.1}
        
        # Hour of day distribution
        hour_weights = [0.1, 0.05, 0.02, 0.02, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.8,
                       1.0, 0.9, 0.7, 0.6, 0.8, 1.2, 1.5, 1.3, 1.0, 0.8, 0.5, 0.2]
        
        # Food items by cuisine
        menu_items = {
            'Italian': [('Margherita Pizza', 18.99), ('Spaghetti Carbonara', 16.99), ('Lasagna', 19.99), 
                       ('Caesar Salad', 12.99), ('Tiramisu', 7.99)],
            'Chinese': [('Kung Pao Chicken', 15.99), ('Sweet & Sour Pork', 16.99), ('Fried Rice', 12.99),
                       ('Spring Rolls', 8.99), ('Hot & Sour Soup', 6.99)],
            'Mexican': [('Chicken Burrito', 13.99), ('Beef Tacos', 11.99), ('Quesadilla', 10.99),
                       ('Guacamole & Chips', 7.99), ('Churros', 5.99)],
            'Indian': [('Butter Chicken', 17.99), ('Biryani', 15.99), ('Naan Bread', 4.99),
                      ('Samosas', 7.99), ('Mango Lassi', 4.99)],
            'Thai': [('Pad Thai', 14.99), ('Green Curry', 16.99), ('Tom Yum Soup', 8.99),
                    ('Spring Rolls', 6.99), ('Mango Sticky Rice', 6.99)]
        }
        
        default_items = [('Main Course', 15.99), ('Side Dish', 8.99), ('Appetizer', 7.99),
                        ('Dessert', 6.99), ('Beverage', 3.99)]
        
        payment_methods = ['credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay']
        order_statuses = ['delivered', 'delivered', 'delivered', 'delivered', 'cancelled']  # 80% delivered
        
        coupon_codes = ['SAVE10', 'WELCOME20', 'FRIDAY15', 'NEWUSER', 'LOYALTY5', None, None, None]
        
        order_id = 1
        order_item_id = 1
        
        for day in range(self.days):
            current_date = self.start_date + timedelta(days=day)
            day_of_week = current_date.weekday()
            day_multiplier = day_multipliers[day_of_week]
            
            orders_today = int(self.config['avg_orders_per_day'] * day_multiplier * random.uniform(0.7, 1.3))
            
            for _ in range(orders_today):
                # Pick random hour based on distribution
                hour = np.random.choice(24, p=np.array(hour_weights)/sum(hour_weights))
                minute = random.randint(0, 59)
                order_time = current_date.replace(hour=hour, minute=minute)
                
                # Select user and restaurant
                user = random.choice(self.users_data)
                restaurant = random.choice([r for r in self.restaurants_data if r['is_active']])
                courier = random.choice([c for c in self.couriers_data if c['is_active']])
                
                # Generate order details
                status = random.choice(order_statuses)
                items_count = np.random.poisson(self.config['avg_items_per_order']) + 1
                
                subtotal = 0.0
                order_items = []
                
                # Get menu items for this restaurant's cuisine
                cuisine_items = menu_items.get(restaurant['cuisine_type'], default_items)
                
                for _ in range(min(items_count, 6)):  # Max 6 items per order
                    item_name, base_price = random.choice(cuisine_items)
                    quantity = random.choices([1, 2, 3], weights=[0.7, 0.25, 0.05])[0]
                    item_price = base_price * random.uniform(0.9, 1.1)  # Price variation
                    
                    order_items.append({
                        'order_item_id': order_item_id,
                        'order_id': order_id,
                        'item_name': item_name,
                        'item_price': round(item_price, 2),
                        'quantity': quantity,
                        'special_instructions': random.choice([None, 'Extra spicy', 'No onions', 'Light sauce']) if random.random() < 0.2 else None,
                        'category': random.choice(['Main', 'Side', 'Appetizer', 'Dessert', 'Beverage'])
                    })
                    
                    subtotal += item_price * quantity
                    order_item_id += 1
                
                # Calculate fees and totals
                delivery_fee = restaurant['delivery_fee']
                service_fee = round(subtotal * 0.1, 2)
                tip_amount = round(subtotal * random.uniform(0.1, 0.25), 2) if random.random() < 0.7 else 0.0
                
                coupon_code = random.choice(coupon_codes)
                discount_amount = 0.0
                if coupon_code:
                    if 'SAVE10' in coupon_code:
                        discount_amount = min(10.0, subtotal * 0.1)
                    elif 'WELCOME20' in coupon_code:
                        discount_amount = min(20.0, subtotal * 0.15)
                    elif 'FRIDAY15' in coupon_code:
                        discount_amount = min(15.0, subtotal * 0.12)
                
                total_amount = subtotal + delivery_fee + service_fee + tip_amount - discount_amount
                
                # Delivery times
                prep_time = restaurant['average_prep_time'] + random.randint(-10, 20)
                delivery_time = random.randint(10, 30)
                
                estimated_delivery = order_time + timedelta(minutes=prep_time + delivery_time)
                actual_delivery = estimated_delivery + timedelta(minutes=random.randint(-5, 15)) if status == 'delivered' else None
                
                order = {
                    'order_id': order_id,
                    'user_id': user['user_id'],
                    'restaurant_id': restaurant['restaurant_id'],
                    'courier_id': courier['courier_id'] if status != 'cancelled' else None,
                    'order_date': order_time,
                    'order_status': status,
                    'subtotal': round(subtotal, 2),
                    'delivery_fee': delivery_fee,
                    'service_fee': service_fee,
                    'tip_amount': tip_amount,
                    'discount_amount': discount_amount,
                    'total_amount': round(total_amount, 2),
                    'payment_method': random.choice(payment_methods),
                    'delivery_address': self.fake.address().replace('\n', ', '),
                    'delivery_latitude': round(float(user['latitude']) + random.uniform(-0.01, 0.01), 6),
                    'delivery_longitude': round(float(user['longitude']) + random.uniform(-0.01, 0.01), 6),
                    'estimated_delivery_time': estimated_delivery,
                    'actual_delivery_time': actual_delivery,
                    'prep_time_minutes': prep_time if status == 'delivered' else None,
                    'delivery_time_minutes': delivery_time if status == 'delivered' else None,
                    'coupon_code': coupon_code
                }
                
                self.orders_data.append(order)
                self.order_items_data.extend(order_items)
                
                # Update user totals
                if status == 'delivered':
                    user['total_orders'] += 1
                    user['lifetime_value'] += total_amount
                
                order_id += 1
    
    def generate_sessions(self):
        """Generate user session data for funnel analysis"""
        session_types = ['app_open', 'search', 'restaurant_view', 'menu_view', 'add_to_cart', 'checkout_start', 'payment_complete']
        device_types = ['ios', 'android', 'web']
        
        session_id = 1
        
        # Generate sessions for orders (successful funnels)
        for order in self.orders_data:
            user = next(u for u in self.users_data if u['user_id'] == order['user_id'])
            
            # Create a funnel leading to the order
            session_time = order['order_date'] - timedelta(minutes=random.randint(5, 30))
            device = random.choice(device_types)
            
            funnel_steps = ['app_open', 'search', 'restaurant_view', 'menu_view', 'add_to_cart', 'checkout_start', 'payment_complete']
            
            for step in funnel_steps:
                session = {
                    'session_id': session_id,
                    'user_id': user['user_id'],
                    'session_date': session_time,
                    'session_type': step,
                    'restaurant_id': order['restaurant_id'] if step in ['restaurant_view', 'menu_view', 'add_to_cart'] else None,
                    'device_type': device,
                    'session_duration_seconds': random.randint(30, 300),
                    'page_views': random.randint(1, 5),
                    'location_latitude': float(user['latitude']),
                    'location_longitude': float(user['longitude'])
                }
                
                self.sessions_data.append(session)
                session_time += timedelta(minutes=random.randint(1, 5))
                session_id += 1
        
        # Generate additional incomplete sessions (dropoffs)
        additional_sessions = len(self.orders_data) * 3  # 3x more sessions than orders
        
        for _ in range(additional_sessions):
            user = random.choice(self.users_data)
            session_date = self.fake.date_time_between(start_date=self.start_date, end_date=self.end_date)
            
            # Random dropoff point
            max_step = random.choices(range(len(session_types)), weights=[0.3, 0.2, 0.2, 0.15, 0.1, 0.04, 0.01])[0]
            
            for i in range(max_step + 1):
                session = {
                    'session_id': session_id,
                    'user_id': user['user_id'],
                    'session_date': session_date,
                    'session_type': session_types[i],
                    'restaurant_id': random.choice(self.restaurants_data)['restaurant_id'] if i >= 2 else None,
                    'device_type': random.choice(device_types),
                    'session_duration_seconds': random.randint(10, 180),
                    'page_views': random.randint(1, 3),
                    'location_latitude': float(user['latitude']),
                    'location_longitude': float(user['longitude'])
                }
                
                self.sessions_data.append(session)
                session_date += timedelta(minutes=random.randint(1, 3))
                session_id += 1
    
    def generate_marketing_events(self):
        """Generate marketing events data"""
        event_types = ['coupon_impression', 'coupon_click', 'coupon_redemption', 'email_sent', 'push_notification_sent', 'push_notification_opened']
        channels = ['email', 'push', 'in_app', 'sms']
        campaigns = ['Summer Sale', 'New User Welcome', 'Weekend Special', 'Loyalty Reward', 'Flash Sale']
        coupon_codes = ['SAVE10', 'WELCOME20', 'FRIDAY15', 'NEWUSER', 'LOYALTY5']
        
        event_id = 1
        
        for user in random.sample(self.users_data, min(8000, len(self.users_data))):  # 8k users get marketing
            num_events = random.randint(1, 10)
            
            for _ in range(num_events):
                event_date = self.fake.date_time_between(start_date=self.start_date, end_date=self.end_date)
                event_type = random.choice(event_types)
                channel = random.choice(channels)
                campaign = random.choice(campaigns)
                
                # Coupon details
                coupon_code = random.choice(coupon_codes) if 'coupon' in event_type else None
                discount_percentage = random.choice([10, 15, 20, 25]) if coupon_code else None
                discount_amount = None
                
                # Find matching order for redemptions
                conversion_order_id = None
                if event_type == 'coupon_redemption':
                    user_orders = [o for o in self.orders_data if o['user_id'] == user['user_id'] and o['coupon_code'] == coupon_code]
                    if user_orders:
                        conversion_order_id = random.choice(user_orders)['order_id']
                
                event = {
                    'event_id': event_id,
                    'user_id': user['user_id'],
                    'event_date': event_date,
                    'event_type': event_type,
                    'campaign_name': campaign,
                    'coupon_code': coupon_code,
                    'discount_percentage': discount_percentage,
                    'discount_amount': discount_amount,
                    'channel': channel,
                    'conversion_order_id': conversion_order_id
                }
                
                self.marketing_events_data.append(event)
                event_id += 1
    
    def generate_reviews(self):
        """Generate reviews data"""
        review_texts = [
            "Great food and fast delivery!", "Food was cold when it arrived", "Amazing flavors, will order again",
            "Driver was very professional", "Food quality has improved", "Late delivery but good food",
            "Perfect portion sizes", "Not worth the price", "Excellent service", "Food was bland",
            "Quick delivery, hot food", "Missing items from my order", "Best restaurant in the area",
            "Rude delivery driver", "Food exceeded expectations", "Long wait time", "Fresh ingredients",
            "Poor packaging, food spilled", "Fantastic customer service", "Average food quality"
        ]
        
        review_id = 1
        
        # Generate reviews for ~40% of delivered orders
        delivered_orders = [o for o in self.orders_data if o['order_status'] == 'delivered']
        review_orders = random.sample(delivered_orders, int(len(delivered_orders) * 0.4))
        
        for order in review_orders:
            review_date = order['actual_delivery_time'] + timedelta(hours=random.randint(1, 72))
            
            # Generate correlated ratings
            base_rating = random.choices([1, 2, 3, 4, 5], weights=[0.02, 0.03, 0.1, 0.35, 0.5])[0]
            food_rating = max(1, min(5, base_rating + random.randint(-1, 1)))
            delivery_rating = max(1, min(5, base_rating + random.randint(-1, 1)))
            overall_rating = round((food_rating + delivery_rating) / 2)
            
            review = {
                'review_id': review_id,
                'order_id': order['order_id'],
                'user_id': order['user_id'],
                'restaurant_id': order['restaurant_id'],
                'courier_id': order['courier_id'],
                'food_rating': food_rating,
                'delivery_rating': delivery_rating,
                'overall_rating': overall_rating,
                'review_text': random.choice(review_texts) if random.random() < 0.7 else None,
                'review_date': review_date,
                'is_verified': 1
            }
            
            self.reviews_data.append(review)
            review_id += 1
    
    def save_to_csv(self, output_dir: str = 'csv_output'):
        """Save all data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        datasets = {
            'users.csv': self.users_data,
            'restaurants.csv': self.restaurants_data,
            'couriers.csv': self.couriers_data,
            'orders.csv': self.orders_data,
            'order_items.csv': self.order_items_data,
            'sessions.csv': self.sessions_data,
            'marketing_events.csv': self.marketing_events_data,
            'reviews.csv': self.reviews_data
        }
        
        for filename, data in datasets.items():
            if data:
                df = pd.DataFrame(data)
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"Saved {len(data)} records to {filepath}")
    
    def save_to_database(self, db_path: str = 'data/datrik.db'):
        """Save all data to SQLite database"""
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create database and tables
        conn = sqlite3.connect(db_path)
        
        # Read and execute schema
        schema_path = 'src/database_schema.sql'
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema (drop existing tables first)
        conn.executescript('PRAGMA foreign_keys = OFF;')
        for table in ['reviews', 'marketing_events', 'sessions', 'order_items', 'orders', 'couriers', 'restaurants', 'users']:
            conn.execute(f'DROP TABLE IF EXISTS {table};')
        
        conn.executescript(schema_sql)
        
        # Insert data
        datasets = [
            ('users', self.users_data),
            ('restaurants', self.restaurants_data),
            ('couriers', self.couriers_data),
            ('orders', self.orders_data),
            ('order_items', self.order_items_data),
            ('sessions', self.sessions_data),
            ('marketing_events', self.marketing_events_data),
            ('reviews', self.reviews_data)
        ]
        
        for table_name, data in datasets:
            if data:
                df = pd.DataFrame(data)
                
                # Convert datetime columns to strings for SQLite
                datetime_columns = df.select_dtypes(include=['datetime64', 'datetime']).columns
                for col in datetime_columns:
                    df[col] = df[col].astype(str)
                
                # Convert any decimal/object columns that might cause issues
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            # Try to convert to float if it's numeric
                            pd.to_numeric(df[col], errors='raise')
                            df[col] = df[col].astype(float)
                        except (ValueError, TypeError):
                            # Keep as string if not numeric
                            df[col] = df[col].astype(str)
                
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Inserted {len(data)} records into {table_name}")
        
        conn.commit()
        conn.close()
        print(f"Database saved to {db_path}")

def main():
    """Main function to generate all data"""
    print("Starting Datrik Data Generation...")
    
    # Initialize generator
    generator = FoodDeliveryDataGenerator(seed=42, days=90)
    
    # Generate all data
    generator.generate_all_data()
    
    # Save to CSV
    print("\nSaving to CSV files...")
    generator.save_to_csv('csv_output')
    
    # Save to database
    print("\nSaving to database...")
    generator.save_to_database('data/datrik.db')
    
    # Print summary
    print("\n" + "="*50)
    print("DATA GENERATION SUMMARY")
    print("="*50)
    print(f"Users: {len(generator.users_data):,}")
    print(f"Restaurants: {len(generator.restaurants_data):,}")
    print(f"Couriers: {len(generator.couriers_data):,}")
    print(f"Orders: {len(generator.orders_data):,}")
    print(f"Order Items: {len(generator.order_items_data):,}")
    print(f"Sessions: {len(generator.sessions_data):,}")
    print(f"Marketing Events: {len(generator.marketing_events_data):,}")
    print(f"Reviews: {len(generator.reviews_data):,}")
    print("="*50)

if __name__ == "__main__":
    main()