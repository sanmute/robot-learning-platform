import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useAuth } from '../App';
import { api } from '../api';

export default function Login() {
  const { login, user } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [error, setError] = useState<string | null>(null);
  const [devLoading, setDevLoading] = useState(false);

  // If already logged in, redirect to dashboard
  useEffect(() => {
    if (user) navigate('/dashboard', { replace: true });
  }, [user, navigate]);

  // Handle OAuth callback — token in query param
  useEffect(() => {
    const token = params.get('token');
    const oauthError = params.get('error');

    if (oauthError) {
      setError('Google sign-in failed. Please try again.');
      return;
    }

    if (token) {
      // Store token, then fetch the user profile
      localStorage.setItem('token', token);
      api
        .getMe()
        .then((u) => {
          login(token, u);
          navigate('/dashboard', { replace: true });
        })
        .catch(() => {
          localStorage.removeItem('token');
          setError('Failed to load your profile. Please try again.');
        });
    }
  }, [params, login, navigate]);

  const handleGoogleLogin = () => {
    window.location.href = '/api/auth/google';
  };

  const handleDevLogin = async () => {
    setDevLoading(true);
    try {
      const { token, user: u } = await api.devLogin();
      login(token, u);
      navigate('/dashboard', { replace: true });
    } catch {
      setError('Dev login failed');
    } finally {
      setDevLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col">
      {/* Top bar */}
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-bold text-brand-700 text-lg">
            🤖 RoboTrain
          </Link>
        </div>
      </header>

      <main className="flex flex-1 items-center justify-center px-4 py-16">
        <div className="card w-full max-w-sm text-center">
          <div className="mb-6">
            <div className="mb-3 text-4xl">🤖</div>
            <h1 className="text-2xl font-bold text-gray-900">Sign in to RoboTrain</h1>
            <p className="mt-1 text-sm text-gray-500">Train your robot in 4 seconds</p>
          </div>

          {error && (
            <div className="mb-4 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          <button
            onClick={handleGoogleLogin}
            className="btn-secondary w-full justify-center gap-3 py-3"
          >
            <svg className="h-5 w-5" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            Continue with Google
          </button>

          {/* Dev login — only shown outside production */}
          {import.meta.env.DEV && (
            <button
              onClick={handleDevLogin}
              disabled={devLoading}
              className="mt-3 btn-secondary w-full justify-center text-xs opacity-60 hover:opacity-100"
            >
              {devLoading ? 'Signing in…' : '🛠 Dev login (local only)'}
            </button>
          )}

          <p className="mt-6 text-xs text-gray-400">
            By signing in you agree to our{' '}
            <span className="underline cursor-pointer">Terms</span> and{' '}
            <span className="underline cursor-pointer">Privacy Policy</span>.
          </p>
        </div>
      </main>
    </div>
  );
}
