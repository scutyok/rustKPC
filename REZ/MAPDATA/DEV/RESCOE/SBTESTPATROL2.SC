Name "sbtestpatrol2"

AbortOnDamage FALSE
AbortOnEnemySighted FALSE
SightRange 256
TimeOut 999.0
FaceMoveDir TRUE
YawRate 180
Walkspeed 600
CheckForEnemyInterval 0.5
TargetPlayer


Loop1:

OnEnemyOutOfRange_Goto DISABLE		
OnTimerExpired_Goto DISABLE	
OnEnemySighted_Goto Attack				
OnDamage_Goto FarAttack			
TimeOut 10.0

StartAnim "Amba"
Wait 0.5
StartAnim "Deflect"
MoveTo path2
StartAnim "Walk"
TurnTo path1
StartAnim "Amba"
Wait 0.5
StartAnim "Deflect"
MoveTo path1
StartAnim "Walk"
TurnTo path2

GoTo Loop1

Attack:
OnDamage_Goto DISABLE
OnEnemySighted_Goto DISABLE
OnEnemyOutOfRange_Goto	Loop1
YawRate		540
TimeOut		2.5

FarAttack:
SightRange	1024.0

if_RandomLessThan25_Goto	Attack_path2
if_RandomLessThan25_Goto	Attack_path1

Attack_path2:
TurnTo path2
Run
StartAnim	"Run"
OnTimerExpired_Goto		Attack_path2a
StartRandomTimer	0.5		0.7

MoveTo		path2

Attack_path2a:
TurnToTarget	0.33	
RangedAttackTarget

if_RandomLessThan50_Goto	Attack_path1

Attack_path1:
TurnTo path1
Run
StartAnim	"Run"
OnTimerExpired_Goto		Attack_path1a
StartRandomTimer	0.5		0.7

MoveTo		path1

Attack_path1a:
TurnToTarget	0.33	
RangedAttackTarget

if_RandomLessThan50_Goto	Attack_path2

GoTo Loop1




