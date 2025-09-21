-- Datrik Food Delivery Database Schema
-- SQLite database schema for realistic food delivery app data

-- Users table
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth DATE,
    gender TEXT CHECK(gender IN ('male', 'female', 'other')),
    registration_date DATETIME NOT NULL,
    city TEXT NOT NULL,
    postal_code TEXT,
    latitude REAL,
    longitude REAL,
    is_active BOOLEAN DEFAULT 1,
    total_orders INTEGER DEFAULT 0,
    lifetime_value REAL DEFAULT 0.0,
    preferred_cuisine TEXT
);

-- Restaurants table
CREATE TABLE restaurants (
    restaurant_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    cuisine_type TEXT NOT NULL,
    rating REAL DEFAULT 0.0,
    total_reviews INTEGER DEFAULT 0,
    address TEXT NOT NULL,
    city TEXT NOT NULL,
    postal_code TEXT,
    latitude REAL,
    longitude REAL,
    phone TEXT,
    opening_time TIME,
    closing_time TIME,
    delivery_fee REAL DEFAULT 2.99,
    minimum_order REAL DEFAULT 15.0,
    average_prep_time INTEGER DEFAULT 30,
    is_active BOOLEAN DEFAULT 1,
    registration_date DATETIME NOT NULL
);

-- Couriers table
CREATE TABLE couriers (
    courier_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    email TEXT,
    vehicle_type TEXT CHECK(vehicle_type IN ('bike', 'scooter', 'car', 'walking')),
    rating REAL DEFAULT 5.0,
    total_deliveries INTEGER DEFAULT 0,
    registration_date DATETIME NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    current_city TEXT
);

-- Orders table
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    restaurant_id INTEGER REFERENCES restaurants(restaurant_id),
    courier_id INTEGER REFERENCES couriers(courier_id),
    order_date DATETIME NOT NULL,
    order_status TEXT CHECK(order_status IN ('pending', 'confirmed', 'preparing', 'ready', 'picked_up', 'delivered', 'cancelled')),
    subtotal REAL NOT NULL,
    delivery_fee REAL NOT NULL,
    service_fee REAL NOT NULL,
    tip_amount REAL DEFAULT 0.0,
    discount_amount REAL DEFAULT 0.0,
    total_amount REAL NOT NULL,
    payment_method TEXT CHECK(payment_method IN ('credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay', 'cash')),
    delivery_address TEXT NOT NULL,
    delivery_latitude REAL,
    delivery_longitude REAL,
    estimated_delivery_time DATETIME,
    actual_delivery_time DATETIME,
    prep_time_minutes INTEGER,
    delivery_time_minutes INTEGER,
    coupon_code TEXT
);

-- Order items table
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    item_name TEXT NOT NULL,
    item_price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    special_instructions TEXT,
    category TEXT
);

-- User sessions table (for funnel analysis)
CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    session_date DATETIME NOT NULL,
    session_type TEXT CHECK(session_type IN ('app_open', 'search', 'restaurant_view', 'menu_view', 'add_to_cart', 'checkout_start', 'payment_complete')),
    restaurant_id INTEGER REFERENCES restaurants(restaurant_id),
    device_type TEXT CHECK(device_type IN ('ios', 'android', 'web')),
    session_duration_seconds INTEGER,
    page_views INTEGER DEFAULT 1,
    location_latitude REAL,
    location_longitude REAL
);

-- Marketing events table
CREATE TABLE marketing_events (
    event_id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    event_date DATETIME NOT NULL,
    event_type TEXT CHECK(event_type IN ('coupon_impression', 'coupon_click', 'coupon_redemption', 'email_sent', 'push_notification_sent', 'push_notification_opened')),
    campaign_name TEXT,
    coupon_code TEXT,
    discount_percentage REAL,
    discount_amount REAL,
    channel TEXT CHECK(channel IN ('email', 'push', 'in_app', 'sms')),
    conversion_order_id INTEGER REFERENCES orders(order_id)
);

-- Reviews table
CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    user_id INTEGER REFERENCES users(user_id),
    restaurant_id INTEGER REFERENCES restaurants(restaurant_id),
    courier_id INTEGER REFERENCES couriers(courier_id),
    food_rating INTEGER CHECK(food_rating >= 1 AND food_rating <= 5),
    delivery_rating INTEGER CHECK(delivery_rating >= 1 AND delivery_rating <= 5),
    overall_rating INTEGER CHECK(overall_rating >= 1 AND overall_rating <= 5),
    review_text TEXT,
    review_date DATETIME NOT NULL,
    is_verified BOOLEAN DEFAULT 1
);

-- Create indexes for performance
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_user ON orders(user_id);
CREATE INDEX idx_orders_restaurant ON orders(restaurant_id);
CREATE INDEX idx_orders_status ON orders(order_status);
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_date ON sessions(session_date);
CREATE INDEX idx_marketing_events_user ON marketing_events(user_id);
CREATE INDEX idx_marketing_events_date ON marketing_events(event_date);
CREATE INDEX idx_reviews_restaurant ON reviews(restaurant_id);
CREATE INDEX idx_reviews_date ON reviews(review_date);