import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { Routes, Route } from 'react-router-dom';
import type { User } from '@robotrain/shared';
import { api, clearToken, setToken } from './api';

import Landing from './pages/Landing';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Train from './pages/Train';
import Results from './pages/Results';
import ProtectedRoute from './components/ProtectedRoute';

// ── Auth context ──────────────────────────────────────────────────────────────

interface AuthCtx {
  user: User | null;
  loading: boolean;
  login: (token: string, user: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthCtx>({
  user: null,
  loading: true,
  login: () => {},
  logout: () => {},
});

export function useAuth(): AuthCtx {
  return useContext(AuthContext);
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      setLoading(false);
      return;
    }
    api
      .getMe()
      .then(setUser)
      .catch(() => clearToken())
      .finally(() => setLoading(false));
  }, []);

  const login = (token: string, u: User) => {
    setToken(token);
    setUser(u);
  };

  const logout = () => {
    clearToken();
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/train"
          element={
            <ProtectedRoute>
              <Train />
            </ProtectedRoute>
          }
        />
        <Route
          path="/results/:jobId"
          element={
            <ProtectedRoute>
              <Results />
            </ProtectedRoute>
          }
        />
      </Routes>
    </AuthContext.Provider>
  );
}

// Re-export so child components can import from one place
export { AuthContext };
export type { AuthCtx };
