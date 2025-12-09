import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, Target, Clock, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import { useAuthStore } from '../stores/authStore';
import { useSystemStore } from '../stores/systemStore';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { Badge } from '../components/ui/Badge';
import { Alert, AlertTitle, AlertDescription } from '../components/ui/Alert';

interface Order {
  id: string;
  symbol: string;
  type: 'market' | 'limit' | 'stop';
  side: 'buy' | 'sell';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'pending' | 'filled' | 'partial' | 'cancelled' | 'rejected';
  filledQuantity: number;
  createdAt: string;
  updatedAt: string;
  userId: string;
}

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
  openedAt: string;
  userId: string;
}

interface TradeHistory {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  realizedPnl: number;
  executedAt: string;
  userId: string;
}

export const Trading: React.FC = () => {
  const { user } = useAuthStore();
  const { socket } = useSystemStore();
  
  const [orders, setOrders] = useState<Order[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [tradeHistory, setTradeHistory] = useState<TradeHistory[]>([]);
  const [marketData, setMarketData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Order form state
  const [orderForm, setOrderForm] = useState({
    symbol: 'BTCUSDT',
    type: 'market' as 'market' | 'limit' | 'stop',
    side: 'buy' as 'buy' | 'sell',
    quantity: 0.1,
    price: 0,
    stopPrice: 0
  });

  useEffect(() => {
    fetchTradingData();
    
    if (socket) {
      socket.on('orderUpdate', handleOrderUpdate);
      socket.on('positionUpdate', handlePositionUpdate);
      socket.on('tradeExecuted', handleTradeExecuted);
      socket.on('marketData', handleMarketData);
    }

    return () => {
      if (socket) {
        socket.off('orderUpdate', handleOrderUpdate);
        socket.off('positionUpdate', handlePositionUpdate);
        socket.off('tradeExecuted', handleTradeExecuted);
        socket.off('marketData', handleMarketData);
      }
    };
  }, [socket]);

  const fetchTradingData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [ordersRes, positionsRes, historyRes, marketRes] = await Promise.all([
        fetch('/api/trading/orders'),
        fetch('/api/trading/positions'),
        fetch('/api/trading/history'),
        fetch('/api/market/data/BTCUSDT')
      ]);

      if (!ordersRes.ok || !positionsRes.ok || !historyRes.ok || !marketRes.ok) {
        throw new Error('Failed to fetch trading data');
      }

      const [ordersData, positionsData, historyData, marketData] = await Promise.all([
        ordersRes.json(),
        positionsRes.json(),
        historyRes.json(),
        marketRes.json()
      ]);

      const mapOrder = (order: any): Order => ({
        id: order.id,
        symbol: order.symbol,
        type: order.type,
        side: order.side,
        quantity: order.quantity,
        price: order.price ?? order.limit_price ?? 0,
        stopPrice: order.stop_price ?? order.stopPrice ?? 0,
        status: order.status,
        filledQuantity: order.filled_quantity ?? order.filledQuantity ?? 0,
        createdAt: order.created_at || order.createdAt || new Date().toISOString(),
        updatedAt: order.updated_at || order.updatedAt || new Date().toISOString(),
        userId: order.user_id || order.userId || 'system',
      });

      const mapPosition = (pos: any): Position => ({
        id: pos.id || pos.position_id || crypto.randomUUID(),
        symbol: pos.symbol,
        side: pos.side,
        quantity: pos.quantity,
        entryPrice: pos.entry_price ?? pos.entryPrice ?? 0,
        currentPrice: pos.current_price ?? pos.currentPrice ?? 0,
        unrealizedPnl: pos.unrealized_pnl ?? pos.unrealizedPnl ?? 0,
        realizedPnl: pos.realized_pnl ?? pos.realizedPnl ?? 0,
        openedAt: pos.opened_at || pos.openedAt || new Date().toISOString(),
        userId: pos.user_id || pos.userId || 'system',
      });

      const mapTrade = (trade: any): TradeHistory => ({
        id: trade.id,
        symbol: trade.symbol,
        side: trade.side,
        quantity: trade.quantity,
        price: trade.price,
        realizedPnl: trade.realized_pnl ?? trade.realizedPnl ?? 0,
        executedAt: trade.executed_at || trade.executedAt || new Date().toISOString(),
        userId: trade.user_id || trade.userId || 'system',
      });

      setOrders((ordersData.orders || ordersData || []).map(mapOrder));
      setPositions((positionsData.positions || positionsData || []).map(mapPosition));
      setTradeHistory((historyData.trades || historyData || []).map(mapTrade));
      setMarketData(marketData.data || marketData || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch trading data');
    } finally {
      setLoading(false);
    }
  };

  const handleOrderUpdate = (order: Order) => {
    setOrders(prev => {
      const existing = prev.find(o => o.id === order.id);
      if (existing) {
        return prev.map(o => o.id === order.id ? order : o);
      }
      return [...prev, order];
    });
  };

  const handlePositionUpdate = (position: Position) => {
    setPositions(prev => {
      const existing = prev.find(p => p.id === position.id);
      if (existing) {
        return prev.map(p => p.id === position.id ? position : p);
      }
      return [...prev, position];
    });
  };

  const handleTradeExecuted = (trade: TradeHistory) => {
    setTradeHistory(prev => [trade, ...prev]);
  };

  const handleMarketData = (data: any[]) => {
    setMarketData(data);
  };

  const submitOrder = async () => {
    try {
      const response = await fetch('/api/trading/orders', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(orderForm)
      });

      if (!response.ok) {
        throw new Error('Failed to submit order');
      }

      const newOrder = await response.json();
      handleOrderUpdate(newOrder);
      
      // Reset form
      setOrderForm({
        symbol: 'BTCUSDT',
        type: 'market',
        side: 'buy',
        quantity: 0.1,
        price: 0,
        stopPrice: 0
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit order');
    }
  };

  const cancelOrder = async (orderId: string) => {
    try {
      const response = await fetch(`/api/trading/orders/${orderId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to cancel order');
      }

      setOrders(prev => prev.map(order => 
        order.id === orderId ? { ...order, status: 'cancelled' } : order
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel order');
    }
  };

  const getOrderStatusColor = (status: string) => {
    switch (status) {
      case 'filled': return 'success';
      case 'partial': return 'warning';
      case 'cancelled': return 'error';
      case 'rejected': return 'error';
      default: return 'default';
    }
  };

  const getPnlColor = (pnl: number) => {
    return pnl >= 0 ? 'text-green-600' : 'text-red-600';
  };

  const totalUnrealizedPnl = positions.reduce((sum, pos) => sum + pos.unrealizedPnl, 0);
  const totalRealizedPnl = positions.reduce((sum, pos) => sum + pos.realizedPnl, 0);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="error" className="mb-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Trading Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Unrealized P&L</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getPnlColor(totalUnrealizedPnl)}`}>
              ${totalUnrealizedPnl.toFixed(2)}
            </div>
            <p className="text-xs text-muted-foreground">
              {totalUnrealizedPnl >= 0 ? '+' : ''}{((totalUnrealizedPnl / (positions.length || 1)) * 100).toFixed(2)}% avg
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Realized P&L</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getPnlColor(totalRealizedPnl)}`}>
              ${totalRealizedPnl.toFixed(2)}
            </div>
            <p className="text-xs text-muted-foreground">
              From {tradeHistory.length} closed trades
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Open Positions</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{positions.length}</div>
            <p className="text-xs text-muted-foreground">
              {positions.filter(p => p.side === 'long').length} long, {positions.filter(p => p.side === 'short').length} short
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Orders</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {orders.filter(o => o.status === 'pending').length}
            </div>
            <p className="text-xs text-muted-foreground">
              {orders.filter(o => o.side === 'buy').length} buy, {orders.filter(o => o.side === 'sell').length} sell
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Order Entry */}
        <Card>
          <CardHeader>
            <CardTitle>Place Order</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Symbol</label>
                <Input
                  value={orderForm.symbol}
                  onChange={(e) => setOrderForm(prev => ({ ...prev, symbol: e.target.value }))}
                  placeholder="BTCUSDT"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Type</label>
                  <Select
                    value={orderForm.type}
                    onChange={(e) => setOrderForm(prev => ({ ...prev, type: e.target.value as any }))}
                  >
                    <option value="market">Market</option>
                    <option value="limit">Limit</option>
                    <option value="stop">Stop</option>
                  </Select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Side</label>
                  <Select
                    value={orderForm.side}
                    onChange={(e) => setOrderForm(prev => ({ ...prev, side: e.target.value as any }))}
                  >
                    <option value="buy">Buy</option>
                    <option value="sell">Sell</option>
                  </Select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Quantity</label>
                <Input
                  type="number"
                  step="0.01"
                  value={orderForm.quantity}
                  onChange={(e) => setOrderForm(prev => ({ ...prev, quantity: parseFloat(e.target.value) }))}
                  placeholder="0.1"
                />
              </div>

              {orderForm.type !== 'market' && (
                <div>
                  <label className="block text-sm font-medium mb-1">
                    {orderForm.type === 'limit' ? 'Limit Price' : 'Stop Price'}
                  </label>
                  <Input
                    type="number"
                    step="0.01"
                    value={orderForm.price || orderForm.stopPrice}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value);
                      setOrderForm(prev => ({
                        ...prev,
                        [prev.type === 'limit' ? 'price' : 'stopPrice']: value
                      }));
                    }}
                    placeholder="50000"
                  />
                </div>
              )}

              <Button 
                onClick={submitOrder}
                className={`w-full ${orderForm.side === 'buy' ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700'}`}
              >
                {orderForm.side === 'buy' ? 'Buy' : 'Sell'} {orderForm.symbol}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Market Chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Market Data - {orderForm.symbol}</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={marketData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="price" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Open Orders */}
      <Card>
        <CardHeader>
          <CardTitle>Open Orders</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Side
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quantity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {orders.map((order) => (
                  <tr key={order.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {order.symbol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {order.type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        order.side === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                      }`}>
                        {order.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {order.quantity}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {order.price || order.stopPrice || 'Market'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <Badge variant={getOrderStatusColor(order.status)}>
                        {order.status}
                      </Badge>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(order.createdAt).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {order.status === 'pending' && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => cancelOrder(order.id)}
                          className="text-red-600 hover:text-red-700"
                        >
                          Cancel
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {orders.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No open orders
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Open Positions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Open Positions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {positions.map((position) => (
                <div key={position.id} className="border rounded-lg p-4">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <h4 className="font-semibold">{position.symbol}</h4>
                      <p className="text-sm text-gray-500">
                        {position.side.toUpperCase()} • {position.quantity}
                      </p>
                    </div>
                    <Badge variant={position.side === 'long' ? 'success' : 'error'}>
                      {position.side}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Entry Price</p>
                      <p className="font-medium">${position.entryPrice.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Current Price</p>
                      <p className="font-medium">${position.currentPrice.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Unrealized P&L</p>
                      <p className={`font-medium ${getPnlColor(position.unrealizedPnl)}`}>
                        ${position.unrealizedPnl.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-500">Realized P&L</p>
                      <p className={`font-medium ${getPnlColor(position.realizedPnl)}`}>
                        ${position.realizedPnl.toFixed(2)}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
              {positions.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No open positions
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Recent Trades */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Trades</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {tradeHistory.slice(0, 10).map((trade) => (
                <div key={trade.id} className="flex justify-between items-center p-3 border rounded-lg">
                  <div>
                    <p className="font-medium">{trade.symbol}</p>
                    <p className="text-sm text-gray-500">
                      {trade.side.toUpperCase()} • {trade.quantity} @ ${trade.price.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className={`font-medium ${getPnlColor(trade.realizedPnl)}`}>
                      ${trade.realizedPnl.toFixed(2)}
                    </p>
                    <p className="text-sm text-gray-500">
                      {new Date(trade.executedAt).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              {tradeHistory.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No recent trades
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};