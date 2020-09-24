#include "config.h"

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/io.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>

#include<bits/stdc++.h> 

#include "jlcxx/jlcxx.hpp"

enum class SolverType {BICGSTAB, RestartedGMRes};

using namespace std;

template <class T, int BS>
class DUNE_ISTL_Solver{
public:
    typedef T type;
    using VectorBlock = Dune::FieldVector<T,BS>;
    using MatrixBlock = Dune::FieldMatrix<T,BS,BS>;
    using BVector = Dune::BlockVector<VectorBlock>;
    using BCRSMat =  Dune::BCRSMatrix<MatrixBlock>;
    using Operator = Dune::MatrixAdapter<BCRSMat,BVector,BVector>;

    // instantiate solvers
    using IterativeSolver = Dune::IterativeSolver<BVector, BVector>;
    using BiCGSTABSolver = Dune::BiCGSTABSolver<BVector>;
    using RestartedGMResSolver = Dune::RestartedGMResSolver<BVector>;

    // instantiate preconditioners
    using Preconditioner = Dune::Preconditioner<BVector, BVector>;
    using SeqILU = Dune::SeqILU<BCRSMat, BVector, BVector>;


    DUNE_ISTL_Solver(const int n)
        : n_(n)
        , solver_type_(SolverType::BICGSTAB)
        , target_reduction_(0.01)
        , max_iter_(50)
        , verbose_(0)
        , gmres_restart_(5)
        , prec_relax_(1.2)
        , ilu_n_(0)
        , ilu_resort_(true)
        , tmp_(0)
    {
        mat_ = std::make_unique<BCRSMat>(n, n, BCRSMat::random);
        fop_ = std::make_unique<Operator>(*mat_);
        rhs_ = std::make_unique<BVector>(n);
        x_ = std::make_unique<BVector>(n);
    }

    int greet(){std::cout << "Welcome to the DUNE ISTL CxxWrapper." << std::endl; return 1;};

    int print_matrix(){
        Dune::printmatrix(std::cout, *mat_, "random", "row");
        return 1;
    }
    
    int print_rhs(){
        Dune::printvector(std::cout, *rhs_, "rhs", "row");
        return 1;
    }

    int construct_matrix(int nnz, int* row_size, int* BI, int* BJ){
        for(int i=0; i<n_; i++)
            mat_->setrowsize(i,row_size[i]);
        mat_->endrowsizes();

        nnz_ = nnz;

        BI_.reset(new int[nnz]);
        BJ_.reset(new int[nnz]);

        for(int i=0; i<nnz; i++){
            mat_->addindex(BI[i], BJ[i]);
            BI_[i] = BI[i];
            BJ_[i] = BJ[i];
        }
        mat_->endindices();
        return 1;
    }

    int set_value_matrix(int I, int J, T* value){
        for(int k=0; k<nnz_; k++){
            (*mat_)[BI_[k]][BJ_[k]][I][J] = value[k];
        }
        return 1;
    }

    int set_value_rhs(int I, T* value){
        for(int k=0; k<nnz_; k++){
            (*rhs_)[k][I] = value[k];
        }
        return 1;
    }

    int solve(){
        // TODO: add more preconditioner
        std::unique_ptr<Preconditioner> prec;
        prec = std::make_unique<SeqILU>(*mat_, ilu_n_, prec_relax_, ilu_resort_);

        std::unique_ptr<IterativeSolver> solver;
        switch (solver_type_)
        {
        case SolverType::BICGSTAB:
            solver = std::make_unique<BiCGSTABSolver>(*fop_, *prec, target_reduction_, max_iter_, verbose_);
            break;
        case SolverType::RestartedGMRes:
            solver = std::make_unique<RestartedGMResSolver>(*fop_, *prec, target_reduction_, gmres_restart_, max_iter_, verbose_);
            break;
        default:
            solver = std::make_unique<BiCGSTABSolver>(*fop_, *prec, target_reduction_, max_iter_, verbose_);
            break;
        }
        
        solver->apply(*x_, *rhs_, res_);
        return 1;
    }

    int get_solution(T* x){
        for(int k=0; k<n_; k++){
            for (int ii=0; ii<BS; ii++){
                x[BS*k+ii] = (*x_)[k][ii];
            }
        }
        return 1;
    }

    void set_solver_type(std::string type) {solver_type_ = string2solvertype(type);}

    std::string get_solver_type() {
        if(solver_type_ == SolverType::BICGSTAB) return "BiCGSTAB";
        if(solver_type_ == SolverType::RestartedGMRes) return "RestartedGMRes";
        return "BiCGSTAB";
    }

    void set_target_reduction(double target_reduction) {target_reduction_ = target_reduction; }
    double get_target_reduction() {return target_reduction_; }

    void set_max_iter(int max_iter) {max_iter_ = max_iter; }
    int get_max_iter() {return max_iter_; }
    
    void set_verbose(int verbose) {verbose_ = verbose; }
    int get_verbose() {return verbose_; }

    void set_ilu_n(int ilu_n) {ilu_n_ = ilu_n; }
    int get_ilu_n() {return ilu_n_; }

    void set_ilu_resort(bool ilu_resort) {ilu_resort_=ilu_resort; }
    bool get_ilu_resort() {return ilu_resort_; }

    void set_prec_relax(double prec_relax) {prec_relax_ = prec_relax; }
    double get_prec_relax() {return prec_relax_; }

    void set_gmres_restart(int gmres_restart) {gmres_restart_ = gmres_restart; }
    int get_gmres_restart() {return gmres_restart_; }

    int get_iterations() {return res_.iterations; }
    bool get_converged() {return res_.converged; }
    double get_reduction() {return res_.reduction; }
    double get_conv_rate() {return res_.conv_rate; }
    double get_elapsed() {return res_.elapsed; }

    int tmp_;
private:
    
    SolverType string2solvertype(std::string type){
        std::transform(type.begin(), type.end(), type.begin(), ::toupper); 
        if (type == "BICGSTAB") return SolverType::BICGSTAB;
        if (type == "RESTARTEDGMRES") return SolverType::RestartedGMRes;
        std::cout << "Unsupported solver " << type << ", using default solver BICGSTB" << std::endl;
        return SolverType::BICGSTAB;
    }
    
    std::unique_ptr<BCRSMat> mat_;
    std::unique_ptr<Operator> fop_;
    std::unique_ptr<BVector> rhs_;
    std::unique_ptr<BVector> x_;
    std::unique_ptr<int[]> BI_;
    std::unique_ptr<int[]> BJ_;

    Dune::InverseOperatorResult res_;

    int n_;
    int nnz_;

    SolverType solver_type_;
    int max_iter_;
    double target_reduction_;
    int gmres_restart_;
    int verbose_;

    // std::string prec_type_;
    // int prec_iter_;
    double prec_relax_;
    int ilu_n_;
    bool ilu_resort_;

};

// Helper to wrap DUNE_ISTL_Solver instances
struct WrapDUNE_ISTL_Solver
{
  template<typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped)
  {
    typedef typename TypeWrapperT::type WrappedT;
    wrapped.template constructor<const int>();
    // Access the module to add a free function
    wrapped.module().method("get_tmp", [](const WrappedT& w) { return w.tmp_; });

    wrapped.method("print_matrix", &WrappedT::print_matrix)
     .method("print_rhs", &WrappedT::print_rhs)
     .method("construct_matrix", &WrappedT::construct_matrix)
     .method("set_value_matrix", &WrappedT::set_value_matrix)
     .method("set_value_rhs", &WrappedT::set_value_rhs)
     .method("solve", &WrappedT::solve)
     .method("get_solution", &WrappedT::get_solution)
     .method("set_solver_type", &WrappedT::set_solver_type)
     .method("get_solver_type", &WrappedT::get_solver_type)
     .method("set_target_reduction", &WrappedT::set_target_reduction)
     .method("get_target_reduction", &WrappedT::get_target_reduction)
     .method("set_max_iter", &WrappedT::set_max_iter)
     .method("get_max_iter", &WrappedT::get_max_iter)
     .method("set_verbose", &WrappedT::set_verbose)
     .method("get_verbose", &WrappedT::get_verbose)
     .method("set_ilu_n", &WrappedT::set_ilu_n)
     .method("get_ilu_n", &WrappedT::get_ilu_n)
     .method("set_ilu_resort", &WrappedT::set_ilu_resort)
     .method("get_ilu_resort", &WrappedT::get_ilu_resort)
     .method("set_prec_relax", &WrappedT::set_prec_relax)
     .method("get_prec_relax", &WrappedT::get_prec_relax)
     .method("set_gmres_restart", &WrappedT::set_gmres_restart)
     .method("get_gmres_restart", &WrappedT::get_gmres_restart)
     .method("get_iterations", &WrappedT::get_iterations)
     .method("get_converged", &WrappedT::get_converged)
     .method("get_reduction", &WrappedT::get_reduction)
     .method("get_conv_rate", &WrappedT::get_conv_rate)
     .method("get_elapsed", &WrappedT::get_elapsed)
     ;
  }
};


namespace jlcxx
{
  template<typename T, int Val>
  struct BuildParameterList<DUNE_ISTL_Solver<T, Val>>
  {
    typedef ParameterList<T, std::integral_constant<int, Val>> type;
  };
} // namespace jlcxx

JLCXX_MODULE define_julia_module(jlcxx::Module& types)
{
  using namespace jlcxx;
  types.add_type<Parametric<TypeVar<1>, TypeVar<2>>>("DUNE_ISTL_Solver")
    .apply< DUNE_ISTL_Solver<double, 1>, 
            DUNE_ISTL_Solver<double, 2>, 
            DUNE_ISTL_Solver<double, 3>,
            DUNE_ISTL_Solver<double, 4>,
            DUNE_ISTL_Solver<float, 1>, 
            DUNE_ISTL_Solver<float, 2>, 
            DUNE_ISTL_Solver<float, 3>,
            DUNE_ISTL_Solver<float, 4>
           >(WrapDUNE_ISTL_Solver());
}


