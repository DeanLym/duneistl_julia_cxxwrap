#include "config.h"

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/io.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>

#include "jlcxx/jlcxx.hpp"

enum class SolverType {BICGSTAB, RestartedGMRes};
enum class PreconditionerType {ILU};

using namespace std;

template <class T, int BS>
class DuneIstlSolver{
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


    DuneIstlSolver(const int n)
        : n_(n)
        , solver_type_(SolverType::BICGSTAB)
        , target_reduction_(0.01)
        , max_iter_(50)
        , verbose_(0)
        , gmres_restart_(5)
        , preconditioner_type_(PreconditionerType::ILU)
        , preconditioner_relax_(1.2)
        , ilu_n_(0)
        , ilu_resort_(true)
        , tmp_(0)
    {
        mat_ = std::make_unique<BCRSMat>(n, n, BCRSMat::random);
        fop_ = std::make_unique<Operator>(*mat_);
        rhs_ = std::make_unique<BVector>(n);
        x_ = std::make_unique<BVector>(n);
    }

    void greet(){std::cout << "Welcome to the DUNE ISTL CxxWrapper." << std::endl;};

    void print_matrix(){
        if (n_ > 100)
            std::cout << "Printing large matrix (n>100) is disabled." << std::endl; 
        else
            Dune::printmatrix(std::cout, *mat_, "random", "");
    }
    
    void print_rhs(){
        if (n_ > 100)
            std::cout << "Printing large vector (n>100) is disabled." << std::endl; 
        else
            Dune::printvector(std::cout, *rhs_, "rhs", "");
    }

    void print_x(){
        if (n_ > 100)
            std::cout << "Printing large vector (n>100) is disabled." << std::endl; 
        else
            Dune::printvector(std::cout, *x_, "x", "");
    }

    void construct_matrix(int nnz, int* row_size, int* BI, int* BJ){
        for(int i=0; i<n_; i++)
            mat_->setrowsize(i,row_size[i]);
        mat_->endrowsizes();

        nnz_ = nnz;

        BI_.reset(new int[nnz]);
        BJ_.reset(new int[nnz]);

        for(int i=0; i<nnz; i++){
            mat_->addindex(BI[i]-1, BJ[i]-1);
            BI_[i] = BI[i];
            BJ_[i] = BJ[i];
        }
        mat_->endindices();
    }

    void add_value_matrix(int nn, int* BI, int* BJ, int I, int J, T* value){
        for(int k=0; k<nn; k++){
            (*mat_)[BI[k]-1][BJ[k]-1][I-1][J-1] += value[k];
        }
    }

    void get_value_matrix(int nn, int*BI, int* BJ, int I, int J, T* value){
        for(int k=0; k<nn; k++){
            value[k] = (*mat_)[BI[k]-1][BJ[k]-1][I-1][J-1];
        }
    }

    void reset_matrix(bool reset_structure){
        if(reset_structure){
            mat_ = std::make_unique<BCRSMat>(n_, n_, BCRSMat::random);
        }else{
            (*mat_) = 0.0;
        }
    }

    void add_value_rhs(int nn, int* BI, int I, T* value){
        for(int k=0; k<nn; k++){
            (*rhs_)[BI[k]-1][I-1] = value[k];
        }
    }

    void get_value_rhs(int nn, int* BI, int I, T* value){
        for(int k=0; k<nn; k++){
            value[k] = (*rhs_)[BI[k]-1][I-1];
        }
    }

    void reset_rhs(){
        (*rhs_) = 0.0;
    }

    void add_value_x(int nn, int* BI, int I, T* value){
        for(int k=0; k<nn; k++){
            (*x_)[BI[k]-1][I-1] = value[k];
        }
    }

    void get_value_x(int nn, int* BI, int I, T* value){
        for(int k=0; k<nn; k++){
            value[k] = (*x_)[BI[k]-1][I-1];
        }
    }

    void reset_x(){
        (*x_) = 0.0;
    }

    void solve(){
        // TODO: add more preconditioner
        std::unique_ptr<Preconditioner> prec;
        prec = std::make_unique<SeqILU>(*mat_, ilu_n_, preconditioner_relax_, ilu_resort_);

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
    }

    void get_solution(T* x){
        for(int k=0; k<n_; k++){
            for (int ii=0; ii<BS; ii++){
                x[BS*k+ii] = (*x_)[k][ii];
            }
        }        
    }

    void set_solver_type(std::string type) {solver_type_ = string2solvertype(type);}

    std::string get_solver_type() {
        if(solver_type_ == SolverType::BICGSTAB) return "BiCGSTAB";
        if(solver_type_ == SolverType::RestartedGMRes) return "RestartedGMRes";
        return "BiCGSTAB";
    }

    void set_preconditioner_type(std::string type){preconditioner_type_ = string2preconditionertype(type);}

    std::string get_preconditioner_type(){
        if(preconditioner_type_ == PreconditionerType::ILU) return "ILU";
        return "ILU";
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

    void set_preconditioner_relax(double preconditioner_relax) {preconditioner_relax_ = preconditioner_relax; }
    double get_preconditioner_relax() {return preconditioner_relax_; }

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

    PreconditionerType string2preconditionertype(std::string type){
        std::transform(type.begin(), type.end(), type.begin(), ::toupper); 
        if (type == "ILU") return PreconditionerType::ILU;
        std::cout << "Unsupported preconditioner " << type << ", using default ILU" << std::endl;
        return PreconditionerType::ILU;
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

    PreconditionerType preconditioner_type_;
    // int preconditioner_iter_;
    double preconditioner_relax_;
    int ilu_n_;
    bool ilu_resort_;

};

// Helper to wrap DuneIstlSolver instances
struct WrapDuneIstlSolver
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
     .method("print_x", &WrappedT::print_x)
     .method("construct_matrix", &WrappedT::construct_matrix)
     .method("add_value_matrix", &WrappedT::add_value_matrix)
     .method("get_value_matrix", &WrappedT::get_value_matrix)
     .method("reset_matrix", &WrappedT::reset_matrix)
     .method("add_value_rhs", &WrappedT::add_value_rhs)
     .method("get_value_rhs", &WrappedT::get_value_rhs)
     .method("reset_rhs", &WrappedT::reset_rhs)
     .method("add_value_x", &WrappedT::add_value_x)
     .method("get_value_x", &WrappedT::get_value_x)
     .method("reset_x", &WrappedT::reset_x)
     .method("solve", &WrappedT::solve)
     .method("get_solution", &WrappedT::get_solution)
     .method("set_solver_type", &WrappedT::set_solver_type)
     .method("get_solver_type", &WrappedT::get_solver_type)
     .method("set_preconditioner_type", &WrappedT::set_preconditioner_type)
     .method("get_preconditioner_type", &WrappedT::get_preconditioner_type)
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
     .method("set_preconditioner_relax", &WrappedT::set_preconditioner_relax)
     .method("get_preconditioner_relax", &WrappedT::get_preconditioner_relax)
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
  struct BuildParameterList<DuneIstlSolver<T, Val>>
  {
    typedef ParameterList<T, std::integral_constant<int, Val>> type;
  };
} // namespace jlcxx

JLCXX_MODULE define_julia_module(jlcxx::Module& types)
{
  using namespace jlcxx;
  types.add_type<Parametric<TypeVar<1>, TypeVar<2>>>("DuneIstlSolver")
    .apply< DuneIstlSolver<double, 1>, 
            DuneIstlSolver<double, 2>, 
            DuneIstlSolver<double, 3>,
            DuneIstlSolver<double, 4>,
            DuneIstlSolver<float, 1>, 
            DuneIstlSolver<float, 2>, 
            DuneIstlSolver<float, 3>,
            DuneIstlSolver<float, 4>
           >(WrapDuneIstlSolver());
}


