"""
Sistema de base de datos para call center
Almacena tickets, clientes, productos y historial
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

from config import DATA_DIR


class CallCenterDatabase:
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = DATA_DIR / 'call_center.db'
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.initialize_database()
    
    def connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def initialize_database(self):
        self.connect()
        
        # Tabla de clientes
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS clientes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cliente_id TEXT UNIQUE NOT NULL,
                nombre TEXT NOT NULL,
                email TEXT,
                telefono TEXT,
                fecha_registro TEXT,
                tipo_cliente TEXT
            )
        ''')
        
        # Tabla de tickets/ordenes
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT UNIQUE NOT NULL,
                cliente_id TEXT NOT NULL,
                tipo TEXT NOT NULL,
                estado TEXT NOT NULL,
                producto TEXT,
                monto REAL,
                descripcion TEXT,
                fecha_creacion TEXT,
                fecha_actualizacion TEXT,
                FOREIGN KEY (cliente_id) REFERENCES clientes (cliente_id)
            )
        ''')
        
        # Tabla de productos
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS productos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                producto_id TEXT UNIQUE NOT NULL,
                nombre TEXT NOT NULL,
                categoria TEXT,
                precio REAL,
                stock INTEGER,
                descripcion TEXT
            )
        ''')
        
        # Tabla de llamadas
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS llamadas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                llamada_id TEXT UNIQUE NOT NULL,
                cliente_id TEXT,
                ticket_id TEXT,
                operador TEXT,
                duracion INTEGER,
                satisfaccion TEXT,
                desempeno_operador TEXT,
                es_venta INTEGER,
                transcripcion TEXT,
                analisis_completo TEXT,
                fecha TEXT,
                FOREIGN KEY (cliente_id) REFERENCES clientes (cliente_id),
                FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
            )
        ''')
        
        self.conn.commit()
        self.populate_sample_data()
        self.close()
    
    def populate_sample_data(self):
        # Verificar si ya hay datos
        self.cursor.execute('SELECT COUNT(*) FROM clientes')
        if self.cursor.fetchone()[0] > 0:
            return
        
        # Clientes de ejemplo
        clientes = [
            ('CR-12345', 'Carlos Rodriguez', 'carlos@email.com', '555-0001', datetime.now().isoformat(), 'Premium'),
            ('CLI-555', 'Maria Garcia', 'maria@email.com', '555-0002', datetime.now().isoformat(), 'Regular'),
            ('CL-789', 'Juan Perez', 'juan@email.com', '555-0003', datetime.now().isoformat(), 'Regular'),
            ('CR-999', 'Ana Martinez', 'ana@email.com', '555-0004', datetime.now().isoformat(), 'Premium'),
            ('CL-444', 'Pedro Lopez', 'pedro@email.com', '555-0005', datetime.now().isoformat(), 'Regular'),
            ('AP-780153', 'Andrew Potter', 'andrew.potter@email.com', '780153', datetime.now().isoformat(), 'Regular'),
            ('AR-917654', 'Austin Robinson', 'austin.robinson@email.com', '917654', datetime.now().isoformat(), 'Premium'),
        ]
        
        self.cursor.executemany('''
            INSERT INTO clientes (cliente_id, nombre, email, telefono, fecha_registro, tipo_cliente)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', clientes)
        
        # Tickets de ejemplo
        tickets = [
            ('556', 'CR-12345', 'Venta', 'Completado', 'Plan Premium', 50.00, 'Compra de plan premium mensual', datetime.now().isoformat(), datetime.now().isoformat()),
            ('557', 'CLI-555', 'Reclamo', 'En Proceso', 'Internet', 0.00, 'Problema con conexion de internet', datetime.now().isoformat(), datetime.now().isoformat()),
            ('558', 'CL-789', 'Consulta', 'Cerrado', 'Plan Basico', 0.00, 'Consulta sobre planes disponibles', datetime.now().isoformat(), datetime.now().isoformat()),
            ('ORD-789', 'CR-999', 'Venta', 'Completado', 'Telefono Modelo X', 299.99, 'Compra de telefono nuevo', datetime.now().isoformat(), datetime.now().isoformat()),
            ('560', 'CL-444', 'Soporte', 'En Proceso', 'Configuracion Email', 0.00, 'Ayuda con configuracion de email', datetime.now().isoformat(), datetime.now().isoformat()),
        ]
        
        self.cursor.executemany('''
            INSERT INTO tickets (ticket_id, cliente_id, tipo, estado, producto, monto, descripcion, fecha_creacion, fecha_actualizacion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tickets)
        
        # Productos de ejemplo
        productos = [
            ('PROD-001', 'Plan Premium', 'Planes', 50.00, 1000, 'Plan premium con internet ilimitado'),
            ('PROD-002', 'Plan Basico', 'Planes', 25.00, 1000, 'Plan basico con 50GB'),
            ('PROD-003', 'Telefono Modelo X', 'Dispositivos', 299.99, 50, 'Telefono inteligente ultima generacion'),
            ('PROD-004', 'Telefono Modelo Y', 'Dispositivos', 199.99, 75, 'Telefono inteligente gama media'),
            ('PROD-005', 'Router WiFi', 'Accesorios', 79.99, 200, 'Router dual band alta velocidad'),
        ]
        
        self.cursor.executemany('''
            INSERT INTO productos (producto_id, nombre, categoria, precio, stock, descripcion)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', productos)
        
        self.conn.commit()
    
    def buscar_cliente(self, cliente_id: str) -> Optional[Dict]:
        self.connect()
        
        self.cursor.execute('''
            SELECT * FROM clientes WHERE cliente_id = ?
        ''', (cliente_id,))
        
        row = self.cursor.fetchone()
        self.close()
        
        if row:
            return dict(row)
        return None
    
    def buscar_ticket(self, ticket_id: str) -> Optional[Dict]:
        self.connect()
        
        self.cursor.execute('''
            SELECT t.*, c.nombre as cliente_nombre, c.email as cliente_email
            FROM tickets t
            LEFT JOIN clientes c ON t.cliente_id = c.cliente_id
            WHERE t.ticket_id = ?
        ''', (ticket_id,))
        
        row = self.cursor.fetchone()
        self.close()
        
        if row:
            return dict(row)
        return None
    
    def buscar_producto(self, nombre: str) -> Optional[Dict]:
        self.connect()
        
        self.cursor.execute('''
            SELECT * FROM productos WHERE nombre LIKE ? OR producto_id = ?
        ''', (f'%{nombre}%', nombre))
        
        row = self.cursor.fetchone()
        self.close()
        
        if row:
            return dict(row)
        return None
    
    def guardar_llamada(self, llamada_data: Dict) -> bool:
        self.connect()
        
        try:
            self.cursor.execute('''
                INSERT INTO llamadas (
                    llamada_id, cliente_id, ticket_id, operador, duracion,
                    satisfaccion, desempeno_operador, es_venta,
                    transcripcion, analisis_completo, fecha
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                llamada_data.get('llamada_id'),
                llamada_data.get('cliente_id'),
                llamada_data.get('ticket_id'),
                llamada_data.get('operador'),
                llamada_data.get('duracion'),
                llamada_data.get('satisfaccion'),
                llamada_data.get('desempeno_operador'),
                1 if llamada_data.get('es_venta') else 0,
                llamada_data.get('transcripcion'),
                json.dumps(llamada_data.get('analisis_completo')),
                datetime.now().isoformat()
            ))
            
            self.conn.commit()
            self.close()
            return True
        
        except Exception as e:
            print(f"Error guardando llamada: {e}")
            self.close()
            return False
    
    def obtener_historial_cliente(self, cliente_id: str) -> List[Dict]:
        self.connect()
        
        self.cursor.execute('''
            SELECT * FROM llamadas WHERE cliente_id = ? ORDER BY fecha DESC
        ''', (cliente_id,))
        
        rows = self.cursor.fetchall()
        self.close()
        
        return [dict(row) for row in rows]
    
    def actualizar_ticket(self, ticket_id: str, updates: Dict) -> bool:
        self.connect()
        
        try:
            set_clause = ', '.join([f'{key} = ?' for key in updates.keys()])
            values = list(updates.values()) + [ticket_id]
            
            self.cursor.execute(f'''
                UPDATE tickets SET {set_clause}, fecha_actualizacion = ?
                WHERE ticket_id = ?
            ''', values + [datetime.now().isoformat()])
            
            self.conn.commit()
            self.close()
            return True
        
        except Exception as e:
            print(f"Error actualizando ticket: {e}")
            self.close()
            return False
    
    def listar_tickets_pendientes(self) -> List[Dict]:
        self.connect()
        
        self.cursor.execute('''
            SELECT t.*, c.nombre as cliente_nombre
            FROM tickets t
            LEFT JOIN clientes c ON t.cliente_id = c.cliente_id
            WHERE t.estado IN ('En Proceso', 'Pendiente')
            ORDER BY t.fecha_creacion DESC
        ''')
        
        rows = self.cursor.fetchall()
        self.close()
        
        return [dict(row) for row in rows]


def test_database():
    db = CallCenterDatabase()
    
    print("\nPrueba de busqueda de cliente:")
    cliente = db.buscar_cliente('CR-12345')
    if cliente:
        print(f"Cliente encontrado: {cliente['nombre']}")
    
    print("\nPrueba de busqueda de ticket:")
    ticket = db.buscar_ticket('556')
    if ticket:
        print(f"Ticket encontrado: {ticket['tipo']} - {ticket['descripcion']}")
    
    print("\nTickets pendientes:")
    pendientes = db.listar_tickets_pendientes()
    for t in pendientes:
        print(f"  {t['ticket_id']}: {t['tipo']} - {t['cliente_nombre']}")


if __name__ == "__main__":
    test_database()