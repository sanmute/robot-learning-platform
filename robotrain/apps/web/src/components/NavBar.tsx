import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../App';

export default function NavBar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <header className="border-b border-gray-200 bg-white">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link to={user ? '/dashboard' : '/'} className="flex items-center gap-2 font-bold text-brand-700 text-lg">
          🤖 <span>RoboTrain</span>
        </Link>

        <div className="flex items-center gap-4">
          {user ? (
            <>
              <Link to="/dashboard" className="text-sm text-gray-600 hover:text-gray-900">
                Dashboard
              </Link>
              <Link to="/train" className="btn-primary text-sm py-1.5 px-4">
                + New Training
              </Link>
              <button
                onClick={handleLogout}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Sign out
              </button>
            </>
          ) : (
            <Link to="/login" className="btn-primary text-sm py-1.5 px-4">
              Sign in
            </Link>
          )}
        </div>
      </nav>
    </header>
  );
}
