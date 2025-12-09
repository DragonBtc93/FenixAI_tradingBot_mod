import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { TrendingUp, TrendingDown, X } from 'lucide-react';
import { formatCurrency } from '@/lib/utils';

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
}

interface PositionsListProps {
  positions: Position[];
  onClosePosition?: (positionId: string) => void;
}

export function PositionsList({ positions, onClosePosition }: PositionsListProps) {
  const getPnlColor = (pnl: number) => pnl >= 0 ? 'text-green-600' : 'text-red-600';
  const getPnlBg = (pnl: number) => pnl >= 0 ? 'bg-green-50' : 'bg-red-50';

  if (positions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Open Positions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            No open positions
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Open Positions</span>
          <Badge variant="outline">{positions.length}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {positions.map((position) => {
            const pnlPercent = ((position.currentPrice - position.entryPrice) / position.entryPrice) * 100;
            const adjustedPnlPercent = position.side === 'short' ? -pnlPercent : pnlPercent;

            return (
              <div
                key={position.id}
                className={`p-4 rounded-lg border ${getPnlBg(position.unrealizedPnl)}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{position.symbol}</span>
                    <Badge variant={position.side === 'long' ? 'success' : 'error'}>
                      {position.side === 'long' ? (
                        <TrendingUp className="w-3 h-3 mr-1" />
                      ) : (
                        <TrendingDown className="w-3 h-3 mr-1" />
                      )}
                      {position.side.toUpperCase()}
                    </Badge>
                  </div>
                  {onClosePosition && (
                    <button
                      onClick={() => onClosePosition(position.id)}
                      className="p-1 hover:bg-gray-200 rounded"
                      title="Close position"
                      aria-label="Close position"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Quantity</span>
                    <div className="font-medium">{position.quantity}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Entry Price</span>
                    <div className="font-medium">{formatCurrency(position.entryPrice)}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Current Price</span>
                    <div className="font-medium">{formatCurrency(position.currentPrice)}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Unrealized P&L</span>
                    <div className={`font-medium ${getPnlColor(position.unrealizedPnl)}`}>
                      {position.unrealizedPnl >= 0 ? '+' : ''}{formatCurrency(position.unrealizedPnl)}
                      <span className="text-xs ml-1">({adjustedPnlPercent.toFixed(2)}%)</span>
                    </div>
                  </div>
                </div>

                <div className="mt-2 text-xs text-gray-400">
                  Opened: {new Date(position.openedAt).toLocaleString()}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

export default PositionsList;
