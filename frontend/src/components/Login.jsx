import React, {useState, useContext} from 'react'
import axios from 'axios'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faSpinner } from '@fortawesome/free-solid-svg-icons'
import { useNavigate } from 'react-router-dom'
import { AuthContext } from '../AuthProvider'

export const Login = () => {
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [loading, setLoading] = useState(false)
    const navigate = useNavigate()
    const [error, setError] = useState("")
    const {isLoggedIn, setIsLoggedIn} = useContext(AuthContext)

    const handleLogin = async (e) => {
        e.preventDefault()
        setLoading(true)
        const userData = {
            username,
            password
        }
        try{
            const response = await axios.post('http://localhost:8000/api/v1/token/', userData)
            localStorage.setItem('accessToken', response.data.access)
            localStorage.setItem('refreshToken', response.data.refresh)
            console.log("Login successful")
            setIsLoggedIn(true)
            navigate('/')
        }catch(err){
            console.log(err.response.data)
            setError("Invalid username or password")
        }finally{
            setLoading(false)
        }
    }

  return (
    <>
     <div className='container'>
      <div className='row justify-content-center'>
        <div className='col-md-6 bg-light-dark p-5 rounded'>
          <h3 className='text-light text-center mb-4'>Logging to our Portal</h3>
          <form onSubmit={handleLogin}>
            <div className="mb-3">
              <input type="text" className='form-control' placeholder='Enter username' value={username} onChange={(e)=>setUsername(e.target.value)}/>
            </div>
            <div className="mb-3">
                <input type="password" className='form-control' placeholder='Enter password' value={password} onChange={(e)=>setPassword(e.target.value)}/>
            </div>
                {error && <div className="text-danger">{error}</div>}
              {loading ?  (
                <button className='btn btn-info d-block mx-auto' type='button' disabled><FontAwesomeIcon icon={faSpinner} spin/>
                  &nbsp;Logging in...
                </button>
              ) : (
                <button type='submit' className='btn btn-info d-block mx-auto'>Login</button>
              )}
          </form>
        </div>
      </div>
     </div>
    </>
  )
}

export default Login